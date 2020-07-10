import sys
import os
import time
import numpy as np

import gym
from gym import spaces
import matplotlib.pyplot as plt

from stable_baselines import logger

from util.softmax_box import SoftmaxBox
import burgers
import weno_coefficients

#TODO: write in a way that does not require ng=order

#Used in step function to extract all the stencils.
def weno_i_stencils_batch(order, q_batch):
    """
    Take a batch of of stencils and approximate the target value in each sub-stencil.
    That is, for each stencil, approximate the target value multiple times using polynomial interpolation
    for different subsets of the stencil.
    This function relies on external coefficients for the polynomial interpolation.
  
    Parameters
    ----------
    order : int
      WENO sub-stencil width.
    q_batch : numpy array
      stencils for each location, shape is [2, grid_width+1, stencil_width].
  
    Returns
    -------
    Return a batch of stencils
  
    """
  
    a_mat = weno_coefficients.a_all[order]

    # These weights are "backwards" in the original formulation.
    # This is easier in the original formulation because we can add the k for our kth stencil to the index,
    # then subtract by a variable amount to get each value, but there's no need to do that here, and flipping
    # it back around makes the expression simpler.
    a_mat = np.flip(a_mat, axis=-1)

    sub_stencil_size = order
    num_sub_stencils = order
    # Adding a row vector and column vector gives us an "outer product" matrix where each row is a sub-stencil.
    sliding_window_indexes = np.arange(sub_stencil_size)[None, :] + np.arange(num_sub_stencils)[:, None]

                                    #[0,:,indexes] causes output to be transposed for some reason
    q_fp_stencil = np.sum(a_mat * q_batch[0][:, sliding_window_indexes], axis=-1)
    q_fm_stencil = np.sum(a_mat * q_batch[1][:, sliding_window_indexes], axis=-1)
      
    return np.array([q_fp_stencil, q_fm_stencil])

#Used in weno_new, below.
def weno_stencils(order, q):
  """
  Compute WENO stencils
  
  Parameters
  ----------
  order : int
    The stencil width.
  q : np array
    Scalar data to reconstruct.

  Returns
  -------
  stencils

  """
  a = weno_coefficients.a_all[order]
  num_points = len(q) - 2 * order
  q_stencils = np.zeros((order, len(q)))
  for i in range(order, num_points+order):   
      for k in range(order):
          for l in range(order):
              q_stencils[k,i] += a[k, l] * q[i+k-l]
  
  return q_stencils

#Used in weno_new, below.
def weno_weights(order, q):
  """
  Compute WENO weights

  Parameters
  ----------
  order : int
    The stencil width.
  q : np array
    Scalar data to reconstruct.

  Returns
  -------
  stencil weights

  """
  C = weno_coefficients.C_all[order]
  sigma = weno_coefficients.sigma_all[order]

  beta = np.zeros((order, len(q)))
  w = np.zeros_like(beta)
  num_points = len(q) - 2 * order
  epsilon = 1e-16
  for i in range(order, num_points+order):
      alpha = np.zeros(order)
      for k in range(order):
          for l in range(order):
              for m in range(l+1):
                  beta[k, i] += sigma[k, l, m] * q[i+k-l] * q[i+k-m]
          alpha[k] = C[k] / (epsilon + beta[k, i]**2)
      w[:, i] = alpha / np.sum(alpha)
  
  return w


#Used in computation of "actual" output
def weno_new(order, q):
  """
  Compute WENO reconstruction

  Parameters
  ----------
  order : int
    Stencil Width.
  q : numpy array
    Scalar data to reconstruct.

  Returns
  -------
  qL: numpy array
    Reconstructed data.

  """
  
  weights = weno_weights(order, q)
  q_stencils = weno_stencils(order, q)
  qL = np.zeros_like(q)
  num_points = len(q) - 2 * order
  for i in range(order, num_points+order):
      qL[i] = np.dot(weights[:, i], q_stencils[:,i])
    
  return qL


def RandomInitialCondition(grid: burgers.Grid1d, 
                           #seed: int = 44,
                           #offset: float = 0.0,
                           amplitude: float = 1.0,
                           k_min: int = 2,
                           k_max: int = 10):
    """ Generate random initial conditions """
    #rs = np.random.RandomState(seed)
    if k_min % 2 == 1:
      k_min += 1
    if k_max % 2 == 2:
      k_max += 1
    step = (k_max-k_min)/2
    k_values = np.arange(k_min, k_max,2)
    #print(k_values)
    k = np.random.choice(k_values, 1)
    b = np.random.uniform(-amplitude, amplitude, 1)
    a = 3.5 - np.abs(b)
    return  a + b * np.sin(k*np.pi*grid.x/(grid.xmax-grid.xmin))  
  
class WENOBurgersEnv(burgers.Simulation, gym.Env):

    metadata = {'render.modes':['human', 'file']}
    
    def __init__(self, grid, C=0.5, weno_order=3, episode_length=300, init_type="sine", record_weights=False):
        self.grid = grid
        self.t = 0.0 # simulation time
        self.C = C   # CFL number
        self.weno_order = weno_order
        self.init_type = init_type

        self.episode_length = episode_length
        self.steps = 0
        self.record_weights = record_weights

        #TODO: transpose so grid length is first dimension
        self.action_space = SoftmaxBox(low=0.0, high=1.0, shape=(2, self.grid.real_length() + 1, weno_order), dtype=np.float64)
        #self.action_space = spaces.Box(low=0.0, high=1.0, shape=(2 * (self.grid.real_length() + 1) * weno_order,),
        #                               dtype=np.float64)
        #self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(2, self.grid.real_length() + 1, 2*weno_order - 1), dtype=np.float64)
        self.observation_space = spaces.Box(low=-1e10, high=1e10,
                                            shape=(2, self.grid.real_length() + 1, 2 * weno_order - 1),
                                            dtype=np.float64)

        self._lines = None

        self.reset()

    def set_record_weights(self, record_weights):
        self.record_weights = record_weights

    def init_cond(self, type="tophat"):
        if type == "smooth_sine":
            self.grid.u = np.sin(2 * np.pi * self.grid.x)
        elif type == "gaussian":
            self.grid.u = 1.0 + np.exp(-60.0*(self.grid.x - 0.5)**2)
        elif type == "random":
            self.grid.u = RandomInitialCondition(self.grid)
        else:
            super().init_cond(type)
        self.grid.uactual[:] = self.grid.u[:] 

    #This is the only thing unique to Burgers, could we make this a general class with this as a parameter?
    def burgers_flux(self, q):
        return 0.5*q**2

    def rk_substep_actual(self):
        
        # get the solution data
        g = self.grid
        
        # apply boundary conditions
        g.fill_BCs()
        
        #comput flux at each point
        f = self.burgers_flux(g.uactual)
        
        # get maximum velocity
        alpha = np.max(abs(g.uactual))
        
        # Lax Friedrichs Flux Splitting
        fp = (f + alpha * g.uactual) / 2
        fm = (f - alpha * g.uactual) / 2
        
        fpr = g.scratch_array()
        fml = g.scratch_array()
        flux = g.scratch_array()
        
        # compute fluxes at the cell edges
        # compute f plus to the right
        fpr[1:] = weno_new(self.weno_order, fp[:-1])
        # compute f minus to the left
        # pass the data in reverse order
        fml[-1::-1] = weno_new(self.weno_order, fm[-1::-1])
        if self.record_weights:
            # Are these slices correct?
            # fp is one on the right, fm is one on the left?
            # (Also note that fp[:-1] is already one smaller on the left.)
            fp_weights = weno_weights(self.weno_order, fp[:-1])[:, self.weno_order:-(self.weno_order)]
            fm_weights = weno_weights(self.weno_order, fm[-1::-1])[:, self.weno_order:-(self.weno_order+1)]
            weights = np.array([fp_weights, fm_weights])
            self._all_actual_weights.append(weights)
        
        # compute flux from fpr and fml
        flux[1:-1] = fpr[1:-1] + fml[1:-1]
        rhs = g.scratch_array()
        
        rhs[1:-1] = 1/g.dx * (flux[1:-1] - flux[2:])
        return rhs

    def Euler_actual(self, dt):
        """
        Compute one time step using explicit Euler time stepping. Performs state transition for a discrete time step using standard WENO.
        Euler time stepping is first order accurate in time. Should be run with a small time step.
        
        Parameters
        ----------
        dt : float
          timestep.
  
        """
        
        g = self.grid
        
        # fill the boundary conditions
        g.fill_BCs()
        
        # RK4
        # Store the data at the start of the step
        u_start = g.uactual.copy()
        k1 = dt * self.rk_substep_actual()
        g.uactual[g.ilo:g.ihi+1] = u_start[g.ilo:g.ihi+1] + k1[g.ilo:g.ihi+1]
   
    def prep_state(self):
        """
        Return state at current time step. Returns fpr and fml vector slices.
  
        Returns
        -------
        state: np array.
            State vector to be sent to the policy. 
            Size: 2 (one for fp and one for fm) BY grid_size BY stencil_size
            Example: order =3, grid = 128
  
        """
        # get the solution data
        g = self.grid
        
        # apply boundary conditions
        g.fill_BCs()
        
        # compute flux at each point
        f = self.burgers_flux(g.u)
        
        # get maximum velocity
        alpha = np.max(abs(g.u))
        
        # Lax Friedrichs Flux Splitting
        fp = (f + alpha * g.u) / 2
        fm = (f - alpha * g.u) / 2
        
        stencil_size = self.weno_order*2 - 1
        num_stencils = g.real_length() + 1
        offset = g.ng - self.weno_order
        # Adding a row vector and column vector gives us an "outer product" matrix where each row is a stencil.
        sliding_window_indexes = offset + np.arange(stencil_size)[None, :] + np.arange(num_stencils)[:, None]
        
        # These are the same stencils for fp and fm, can that be right?
        # Still getting lost in the exact formulation.
        # Maybe the fp_stencils are shifted by -1/2 and the fm_stencils by +1/2?
        # Seems to work anyway.
        fp_stencils = fp[sliding_window_indexes]
        fm_stencils = fm[sliding_window_indexes]

        state = np.array([fp_stencils, fm_stencils])

        #TODO: transpose state so nx is first dimension. This makes it the batch dimension.
        
        # save this state so that we can use it to compute next state
        self.current_state = state
        
        return state
    
    def reset(self):
        """
        Reset the environment.

        Returns
        -------
        Initial state.

        """
      
        self.t = 0.0
        self.steps = 0
        if self.init_type is None:
            self.init_cond("random")
        else:
            self.init_cond(self.init_type)

        self._all_learned_weights = []
        self._all_actual_weights = []
        
        return self.prep_state()
      
      
    def step(self, action):
        """
        Perform a single time step.

        Parameters
        ----------
        action : np array
          Weights for reconstructing WENO weights. This will be multiplied with the output from weno_stencils
          size: grid-points X order X 2
          Note: at each i+1/2 location we have an fpl and fmr.

        Returns
        -------
        state: np array.
          solution predicted using action
        reward: float
          reward for current action
        done: boolean
          boolean signifying end of episode
        info : dictionary
          not passing anything now
        """

        # Record weights if that mode is enabled.
        if self.record_weights:
            self._all_learned_weights.append(action)
        
        done = False
        g = self.grid
        
        # Store the data at the start of the time step      
        u_copy = g.u.copy()
        
        # passing self.C will cause this to take variable time steps
        # for now work with constant time step = 0.0005
        dt = self.timestep()
    
        state = self.current_state
        
        q_stencils = weno_i_stencils_batch(self.weno_order, state)
       
        fpr, fml = np.sum(action * q_stencils, axis=-1)
    
        flux = fml + fpr
    
        rhs = (flux[:-1] - flux[1:]) / g.dx
        
        u_copy[self.grid.ng:-self.grid.ng] += dt * rhs
        g.u = u_copy
        
        #update the solution time
        self.t += dt

        # Calculate new state.
        state = self.prep_state()

        self.steps += 1
        if self.steps >= self.episode_length:
          done = True
        
        #compute reward
        # Error-based reward.
        reward = 0.0
        self.Euler_actual(dt)

        # max error
        #error = np.max(np.abs(g.u[g.ilo:g.ihi+1]-g.uactual[g.ilo:g.ihi+1]))

        # square error vector
        # We need a reward at each interface, so take one extra cell on either side
        # and compute the average error between each pair.
        error = (g.u[g.ilo-1:g.ihi+2]-g.uactual[g.ilo-1:g.ihi+2])**2
        avg_error = (error[:-1] + error[1:]) / 2

        #reward = -np.log(error)
        reward = -np.arctan(avg_error)
        
        # should this reward be clipped?
        #if reward < 10:
          #reward = 0

        # Conservation-based reward.
        #reward = -np.log(np.sum(rhs[g.ilo:g.ihi+1]))

        # Give a penalty and end the episode if we're way off.
        if np.max(state) > 1e10 or np.isnan(np.max(state)):
            reward -= 100
            done = True

        return state, reward, done, {}

    def render(self, mode='file', **kwargs):
        if mode == "file":
            self.save_plot(**kwargs)
        else:
            print("Render mode: \"" + str(mode)+ "\" not currently implemented.")
            sys.exit(0)


    def save_plot(self, suffix=None):
        if self._lines is None:
            self._step_precision = int(np.ceil(np.log(self.episode_length) / np.log(10)))
            self._lines = []
        else:
            for line in self._lines:
                line.remove()
            self._lines = []


        full_x = self.grid.x
        real_x = full_x[self.grid.ilo:self.grid.ihi+1]

        full_actual = self.grid.uactual
        real_actual = full_actual[self.grid.ilo:self.grid.ihi+1]
        self._lines += plt.plot(real_x, real_actual, ls='-', color='b', label="true")
        
        full_learned = self.grid.u
        real_learned = full_learned[self.grid.ilo:self.grid.ihi+1]
        self._lines += plt.plot(real_x, real_learned, ls='-', color='k', label="predict")

        show_ghost = True
        # The ghost arrays slice off one real point so the line connects to the real points.
        # Leave off labels for these lines so they don't show up in the legend.
        if show_ghost:
            ghost_x_left = full_x[:self.grid.ilo+1]
            ghost_x_right = full_x[self.grid.ihi:]
            ghost_actual_left = full_actual[:self.grid.ilo+1]
            ghost_actual_right = full_actual[self.grid.ihi:]
            ghost_blue = "#80b0ff"
            self._lines += plt.plot(ghost_x_left, ghost_actual_left, ls='-', color=ghost_blue)
            self._lines += plt.plot(ghost_x_right, ghost_actual_right, ls='-', color=ghost_blue)
            ghost_learned_left = full_learned[:self.grid.ilo+1]
            ghost_learned_right = full_learned[self.grid.ihi:]
            ghost_black = "#808080" #ie grey
            self._lines += plt.plot(ghost_x_left, ghost_learned_left, ls='-', color=ghost_black)
            self._lines += plt.plot(ghost_x_right, ghost_learned_right, ls='-', color=ghost_black)

        plt.legend()

        ax = plt.gca()

        # Recalculate automatic axis scaling.
        ax.relim()
        ax.autoscale_view()

        log_dir = logger.get_dir()
        if suffix is None:
            suffix = ("_t{:0" + str(self._step_precision) + "}").format(self.steps)
        filename = 'burgers' + suffix + '.png'
        filename = os.path.join(log_dir, filename)
        plt.savefig(filename)
        print('Saved plot to ' + filename + '.')

    def plot_weights(self):
        vertical_size = 1.0 + 4.0*(2)
        horizontal_size = 1.0 + 4.0*(self.weno_order)
        fig = plt.figure("weights", figsize=(horizontal_size, vertical_size))

        learned_weights = np.array(self._all_learned_weights)
        actual_weights = np.array(self._all_actual_weights)

        # Transpose to put time dimension at the end.
        # Also fix difference between learned and actual shape.
        learned_weights = learned_weights.transpose((1, 3, 2, 0))
        actual_weights = actual_weights.transpose((1, 2, 3, 0))

        # Combine space and time dimensions.
        learned_weights = learned_weights.reshape((2, self.weno_order, -1))
        actual_weights = actual_weights.reshape((2, self.weno_order, -1))

        for row, sign in zip((0,1), ("-", "+")):
            for col, offset in zip(range(self.weno_order), (-1,0,1)): #TODO: figure out offset with order!=3
                label = "f^" + sign + "[" + str(offset) + "]"
                ax = fig.add_subplot(2, self.weno_order, (1 + row*self.weno_order + col), label=label)

                ax.set_xlabel("learned")
                ax.set_ylabel("actual")
                ax.plot(learned_weights[row,col,:], actual_weights[row,col,:], '.')

        log_dir = logger.get_dir()
        filename = 'burgers_weights_comparison.png'
        filename = os.path.join(log_dir, filename)
        plt.savefig(filename)
        print('Saved plot to ' + filename + '.')


    def close(self):
        # Delete references for easier garbage collection.
        self.grid = None

    def seed(self):
        # The official Env class has this as part of its interface, but I don't think we need it. Better to set the seed at the experiment level then the environment level
        pass
