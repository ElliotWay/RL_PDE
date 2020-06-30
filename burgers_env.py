import time
import numpy as np

import gym
from gym import spaces

import burgers
import weno_coefficients
       
#Used in weno_i_stencils_batch below.
def weno_i_stencils(order, q):
  """
  Get sub-stencil approximations at a given location in the grid.
  That is, approximate the target value multiple times using polynomial interpolation for different sets of points.
  This function relies on external coefficients for the polynomial interpolation.

  Parameters
  ----------
  order : int
    The sub-stencil width.
  q : np array
    flux vector stencil from which to create sub-stencils.

  Returns
  -------
  np array of stencils in q each of size order

  """
  a = weno_coefficients.a_all[order]
  q_stencils = np.zeros(order)
  #TODO: remove these loops; generate matrix for q and do matmul
  for k in range(order):
    for l in range(order):
      q_stencils[k] += a[k, l] * q[order-1+k-l]
      
  return q_stencils

#Used in step function to extract all the stencils.
def weno_i_stencils_batch(order, q_batch):
  """
  Take a batch of pieces of state and returns the stencil values.

  Parameters
  ----------
  order : int
    WENO sub-stencil width.
  q_batch : np array
    flux vectors for each location, shape is [2, grid_width, stencil_width].

  Returns
  -------
  Return a batch of stencils

  """
  
  #TODO: vectorize properly, possibly combine with weno_i_stencils
  q_fp_stencil = []
  q_fm_stencil = []
  batch_size = q_batch.shape[1]
  for i in range(batch_size):
    q_fp_stencil.append(weno_i_stencils(order, q_batch[0,i,:]))
    q_fm_stencil.append(weno_i_stencils(order, q_batch[1,i,:]))
    
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
    
    def __init__(self, grid, C=0.5, weno_order=3, episode_length=300, init_type="sine"):
        self.grid = grid
        self.t = 0.0 # simulation time
        self.C = C   # CFL number
        self.weno_order = weno_order
        self.init_type = init_type

        self.episode_length = episode_length
        self.steps = 0

        #TODO: transpose so grid length is first dimension
        self.action_space = spaces.Box(low=0.0, high=1.0, shape=(2, self.grid.real_length() + 1, weno_order), dtype=np.float64)
        self.state_space = spaces.Box(low=-np.inf, high=np.inf, shape=(2, self.grid.real_length() + 1, 2*weno_order - 1), dtype=np.float64)
        
        #What did this do?
        #self.tmax_episode = 0.02*(self.grid.xmax - self.grid.xmin)/1.0

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

      Returns
      -------
      Return state after taking one time step.

      """
      
      g = self.grid
      
      # fill the boundary conditions
      g.fill_BCs()
      
      # RK4
      # Store the data at the start of the step
      u_start = g.uactual.copy()
      k1 = dt * self.rk_substep_actual()
      g.uactual = u_start + k1
   
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
      
      w_o = self.weno_order
      fp_stencils = []
      fm_stencils = []
      
      for i in range(w_o, g.full_length()-w_o):
        fp_stencil = fp[i-w_o+1:i+w_o]
        fp_stencils.append(fp_stencil)
        fm_stencil = fm[i-w_o:i+w_o-1]
        fm_stencils.append(fm_stencil)
        
      state = np.array([fp_stencils, fm_stencils])
      
      #TODO: transpose state so nx is first dimension. This makes it the batch dimension.
      
      # save this state so that we can use it to compute next state
      self.current_state = state
      
      return state
    
    def reset(self):
        """
        Reset the environment.Tasks here:
          Reset initial conditions.
          Set time to 0.

        Returns
        -------
        None.

        """
      
        self.t = 0.0
        self.steps = 0
        if self.init_type is None:
            self.init_cond("random")
        else:
            self.init_cond(self.init_type)
        
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
          scalar value - reward for current action
        done: boolean
          boolean signifying end of episode
        info : dictionary
          not passing anything now
        """

        done = False
        g = self.grid
        
        # Store the data at the start of the time step      
        u_start = g.u.copy()
        
        # passing self.C will cause this to take variable time steps
        # for now work with constant time step = 0.0005
        dt = self.timestep()
    
        state = self.current_state
        
        q_stencils = weno_i_stencils_batch(self.weno_order, state)
       
        fpr, fml = np.sum(action * q_stencils, axis=-1)
    
        # Is this accurate?
        # fpr and fml are 'offset' by one because fpr has an implied +1/2 and fml has an implied -1/2
        # Adding them together, the resulting flux has the implied -1/2.
        flux = np.zeros(g.real_length()+3)
        flux[:-1] = fml
        flux[1:] += fpr
        # Why is it not this instead?
        # TODO: understand why it's not this and change it so this works.
        #flux = np.zeros(g.real_length()+1)
        #flux = fml
        #flux += fpr
    
        # rhs must be "full sized" so it can be added directly to the grid's array.
        # Also, note that rhs indexes line up correctly again (albeit offset by the # of left ghost points).
        rhs = np.zeros(g.full_length())
        rhs[self.weno_order:-self.weno_order] = 1/g.dx * (flux[:-1] - flux[1:])
        
        k1 = dt * rhs
        g.u = u_start + k1
        
        #update the solution time
        self.t += dt

        self.steps += 1
        if self.steps >= self.episode_length:
          done = True
        
        #compute reward
        # Error-based reward.
        reward = 0.0
        self.Euler_actual(dt)
        
        error = np.max(np.abs(g.u[g.ilo:g.ihi+1]-g.uactual[g.ilo:g.ihi+1]))
        reward = -np.log(error)
        
        # should this reward be clipped?
        if reward < 10:
          reward = 0

        # Conservation-based reward.
        #reward = -np.log(np.sum(rhs[g.ilo:g.ihi+1]))

        return self.prep_state(), reward, done, None

    def render(self, mode='file'):
        print("Render not currently implemented")
        #TODO implement

    def close(self):
        #Delete references for easier garbage collection.
        self.grid = None

    def seed(self):
        #The official Env class has this as part of its interface, but I don't think we need it. Better to set the seed at the experiment level then the environment level
        pass
