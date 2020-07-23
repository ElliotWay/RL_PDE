import os
import sys

import gym
import matplotlib.pyplot as plt
import numpy as np
from gym import spaces
from stable_baselines import logger

import burgers
import weno_coefficients
from util.softmax_box import SoftmaxBox
from util.misc import create_stencil_indexes


# TODO: write in a way that does not require ng=order

# Used in step function to extract all the stencils.
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

    # [0,:,indexes] causes output to be transposed for some reason
    q_fp_stencil = np.sum(a_mat * q_batch[0][:, sliding_window_indexes], axis=-1)
    q_fm_stencil = np.sum(a_mat * q_batch[1][:, sliding_window_indexes], axis=-1)

    return np.array([q_fp_stencil, q_fm_stencil])


# Used in weno_new, below.
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
    for i in range(order, num_points + order):
        for k in range(order):
            for l in range(order):
                q_stencils[k, i] += a[k, l] * q[i + k - l]

    return q_stencils

# Used in step to calculate actual weights to compare learned and weno weights.
def weno_weights_batch(order, q_batch):
    """
    Compute WENO weights

    Parameters
    ----------
    order : int
      The stencil width.
    q : np array
      Batch of flux stencils ((nx+1) X (2*order-1))

    Returns
    -------
    stencil weights ((nx+1) X (order))

    """
    C = weno_coefficients.C_all[order]
    sigma = weno_coefficients.sigma_all[order]

    num_points = q_batch.shape[0]
    beta = np.zeros((num_points, order))
    w = np.zeros_like(beta)
    epsilon = 1e-16
    for i in range(num_points):
        alpha = np.zeros(order)
        for k in range(order):
            for l in range(order):
                for m in range(l + 1):
                    beta[i, k] += sigma[k, l, m] * q_batch[i][(order - 1) + k - l] * q_batch[i][(order - 1) + k - m]
            alpha[k] = C[k] / (epsilon + beta[i, k] ** 2)
        w[i, :] = alpha / np.sum(alpha)

    return w


# Used in weno_new, below.
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
    for i in range(order, num_points + order):
        alpha = np.zeros(order)
        for k in range(order):
            for l in range(order):
                for m in range(l + 1):
                    beta[k, i] += sigma[k, l, m] * q[i + k - l] * q[i + k - m]
            alpha[k] = C[k] / (epsilon + beta[k, i] ** 2)
        w[:, i] = alpha / np.sum(alpha)

    return w


# Used in computation of "actual" output
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
    for i in range(order, num_points + order):
        qL[i] = np.dot(weights[:, i], q_stencils[:, i])

    return qL


def RandomInitialCondition(grid: burgers.Grid1d,
                           # seed: int = 44,
                           # offset: float = 0.0,
                           amplitude: float = 1.0,
                           k_min: int = 2,
                           k_max: int = 10):
    """ Generate random initial conditions """
    # rs = np.random.RandomState(seed)
    if k_min % 2 == 1:
        k_min += 1
    if k_max % 2 == 2:
        k_max += 1
    step = (k_max - k_min) / 2
    k_values = np.arange(k_min, k_max, 2)
    # print(k_values)
    k = np.random.choice(k_values, 1)
    b = np.random.uniform(-amplitude, amplitude, 1)
    a = 3.5 - np.abs(b)
    return a + b * np.sin(k * np.pi * grid.x / (grid.xmax - grid.xmin))


class WENOBurgersEnv(burgers.Simulation, gym.Env):
    metadata = {'render.modes': ['human', 'file']}

    def __init__(self, grid, C=0.5, weno_order=3, episode_length=300, init_type="sine", record_weights=False):
        self.grid = grid
        self.t = 0.0  # simulation time
        self.C = C  # CFL number
        self.weno_order = weno_order
        self.init_type = init_type

        self.episode_length = episode_length
        self.steps = 0
        self.record_weights = record_weights

        # TODO: transpose so grid length is first dimension
        self.action_space = SoftmaxBox(low=0.0, high=1.0, shape=(2, self.grid.real_length() + 1, weno_order),
                                       dtype=np.float64)
        # self.action_space = spaces.Box(low=0.0, high=1.0, shape=(2 * (self.grid.real_length() + 1) * weno_order,),
        #                               dtype=np.float64)
        # self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(2, self.grid.real_length() + 1, 2*weno_order - 1), dtype=np.float64)
        self.observation_space = spaces.Box(low=-1e10, high=1e10,
                                            shape=(2, self.grid.real_length() + 1, 2 * weno_order - 1),
                                            dtype=np.float64)

        self._solution_axes = None

        self._init_schedule_index = 0
        self._init_schedule = ["smooth_rare", "smooth_sine", "random", "rarefaction", "accelshock"]
        self._init_sample_types = self._init_schedule
        self._init_sample_probs = [0.2, 0.2, 0.2, 0.2, 0.2]

        #self.reset()

    def set_record_weights(self, record_weights):
        self.record_weights = record_weights

    def init_cond(self, type="tophat"):
        if type == "schedule":
            type = self._init_schedule[self._init_schedule_index]
            self._init_schedule_index = (self._init_schedule_index + 1) % len(self._init_schedule)
        elif type == "sample":
            type = np.random.choice(self._init_sample_types, p=self._init_sample_probs)

        print("Using initial condition " + type)

        if type == "smooth_sine":
            self.grid.set_bc_type("periodic")
            self.grid.u = np.sin(2 * np.pi * self.grid.x)
        elif type == "gaussian":
            self.grid.set_bc_type("periodic")
            self.grid.u = 1.0 + np.exp(-60.0 * (self.grid.x - 0.5) ** 2)
        elif type == "random":
            self.grid.set_bc_type("periodic")
            self.grid.u = RandomInitialCondition(self.grid)
        elif type == "smooth_rare":
            self.grid.set_bc_type("outflow")
            k = np.random.uniform(20, 100)
            self.grid.u = np.tanh(k * (self.grid.x - 0.5))
        elif type == "accelshock":
            self.grid.set_bc_type("outflow")
            index = self.grid.x > 0.25
            self.grid.u[:] = 3
            self.grid.u[index] = 3 * (self.grid.x[index] - 1)
        else:
            super().init_cond(type)
        self.grid.uactual[:] = self.grid.u[:]

        # This is the only thing unique to Burgers, could we make this a general class with this as a parameter?

    def burgers_flux(self, q):
        return 0.5 * q ** 2

    def rk_substep_actual(self):

        # get the solution data
        g = self.grid

        # apply boundary conditions
        g.fill_BCs()

        # comput flux at each point
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

        rhs[1:-1] = 1 / g.dx * (flux[1:-1] - flux[2:])
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
        g.uactual[g.ilo:g.ihi + 1] = u_start[g.ilo:g.ihi + 1] + k1[g.ilo:g.ihi + 1]

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

        stencil_size = self.weno_order * 2 - 1
        num_stencils = g.real_length() + 1
        offset = g.ng - self.weno_order
        # Adding a row vector and column vector gives us an "outer product" matrix where each row is a stencil.
        fp_stencil_indexes = offset + np.arange(stencil_size)[None, :] + np.arange(num_stencils)[:, None]
        fm_stencil_indexes = fp_stencil_indexes + 1

        fp_stencils = fp[fp_stencil_indexes]
        fm_stencils = fm[fm_stencil_indexes]

        # Flip fm stencils. Not sure how I missed this originally?
        fm_stencils = np.flip(fm_stencils, axis=-1)

        state = np.array([fp_stencils, fm_stencils])

        # TODO: transpose state so nx is first dimension. This makes it the batch dimension.

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
        self._all_weno_weights = []

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

        state = self.current_state

        # Record weights if that mode is enabled.
        if self.record_weights:
            self._all_learned_weights.append(action)
            weno_weights_fp = weno_weights_batch(self.weno_order, state[0])
            weno_weights_fm = weno_weights_batch(self.weno_order, state[1])
            weno_weights = np.array([weno_weights_fp, weno_weights_fm])
            self._all_weno_weights.append(weno_weights)

        done = False
        g = self.grid

        # Store the data at the start of the time step      
        u_copy = g.u.copy()

        # passing self.C will cause this to take variable time steps
        # for now work with constant time step = 0.0005
        dt = self.timestep()

        q_stencils = weno_i_stencils_batch(self.weno_order, state)

        fpr, fml = np.sum(action * q_stencils, axis=-1)

        flux = fml + fpr

        rhs = (flux[:-1] - flux[1:]) / g.dx

        u_copy[self.grid.ng:-self.grid.ng] += dt * rhs
        g.u = u_copy

        # update the solution time
        self.t += dt

        # Calculate new state.
        state = self.prep_state()

        self.steps += 1
        if self.steps >= self.episode_length:
            done = True

        # compute reward
        # Error-based reward.
        reward = 0.0
        self.Euler_actual(dt)

        # max error
        # error = np.max(np.abs(g.u[g.ilo:g.ihi+1]-g.uactual[g.ilo:g.ihi+1]))

        # avg square error between cells
        # TODO calculate error for each interface in a way that doesn't go into the ghost cells
        # error = (g.u[g.ilo:g.ihi]-g.uactual[g.ilo:g.ihi])**2
        # error = (error[:-1] + error[1:]) / 2

        # square error on right (misses reward for rightmost interface)
        error = (g.u[g.ilo:g.ihi] - g.uactual[g.ilo:g.ihi]) ** 2

        # Clip tiny errors.
        #error[error < 0.001] = 0
        # Enhance extreme errors.
        #error[error > 0.1] *= 10

        # Reward as function of the errors in the stencil.
        # max error across stencil
        #error = (g.uactual - g.u)
        #stencil_indexes = create_stencil_indexes(stencil_size=(self.weno_order*2-1), num_stencils=(g.ihi - g.ilo + 2), offset=(g.ng - self.weno_order))
        #error_stencils = error[stencil_indexes]
        #error = np.amax(np.abs(error_stencils), axis=-1)
        #error = np.sqrt(np.sum(error_stencils**2, axis=-1))

        # Squash error.
        reward = -np.arctan(error)
        max_penalty = np.pi / 2
        #reward = -error
        #max_penalty = 1e7


        # Conservation-based reward.
        # reward = -np.log(np.sum(rhs[g.ilo:g.ihi+1]))

        # Give a penalty and end the episode if we're way off.
        if np.max(state) > 1e7 or np.isnan(np.max(state)):
            reward -= max_penalty / 2 * self.steps
            done = True

        return state, reward, done, {}

    def render(self, mode='file', **kwargs):
        if mode == "file":
            self.save_plot(**kwargs)
        else:
            print("Render mode: \"" + str(mode) + "\" not currently implemented.")
            sys.exit(0)

    def save_plot(self, suffix=None, fixed_axes=False):
        fig = plt.figure()

        ax = plt.gca()

        full_x = self.grid.x
        real_x = full_x[self.grid.ilo:self.grid.ihi + 1]

        full_actual = self.grid.uactual
        real_actual = full_actual[self.grid.ilo:self.grid.ihi + 1]
        plt.plot(real_x, real_actual, ls='-', color='b', label="WENO")

        full_learned = self.grid.u
        real_learned = full_learned[self.grid.ilo:self.grid.ihi + 1]
        plt.plot(real_x, real_learned, ls='-', color='k', label="RL")

        show_ghost = True
        # The ghost arrays slice off one real point so the line connects to the real points.
        # Leave off labels for these lines so they don't show up in the legend.
        if show_ghost:
            ghost_x_left = full_x[:self.grid.ilo + 1]
            ghost_x_right = full_x[self.grid.ihi:]
            ghost_actual_left = full_actual[:self.grid.ilo + 1]
            ghost_actual_right = full_actual[self.grid.ihi:]
            ghost_blue = "#80b0ff"
            plt.plot(ghost_x_left, ghost_actual_left, ls='-', color=ghost_blue)
            plt.plot(ghost_x_right, ghost_actual_right, ls='-', color=ghost_blue)
            ghost_learned_left = full_learned[:self.grid.ilo + 1]
            ghost_learned_right = full_learned[self.grid.ihi:]
            ghost_black = "#808080"  # ie grey
            plt.plot(ghost_x_left, ghost_learned_left, ls='-', color=ghost_black)
            plt.plot(ghost_x_right, ghost_learned_right, ls='-', color=ghost_black)

        plt.legend()

        ax = plt.gca()

        # Recalculate automatic axis scaling.
        #ax.relim()
        #ax.autoscale_view()

        ax.set_title("step=" + str(self.steps))

        if fixed_axes:
            if self._solution_axes is None:
                self._solution_axes = (ax.get_xlim(), ax.get_ylim())
            else:
                xlim, ylim = self._solution_axes
                ax.set_xlim(xlim)
                ax.set_ylim(ylim)

        log_dir = logger.get_dir()
        if suffix is None:
            step_precision = int(np.ceil(np.log(self.episode_length) / np.log(10)))
            suffix = ("_t{:0" + str(step_precision) + "}").format(self.steps)
        filename = 'burgers' + suffix + '.png'
        filename = os.path.join(log_dir, filename)
        plt.savefig(filename)
        print('Saved plot to ' + filename + '.')

        plt.close(fig)

    def plot_weights(self, timestep=None, location=None):
        assert(len(self._all_learned_weights) == len(self._all_weno_weights))
        if len(self._all_learned_weights) == 0:
            print("Can't plot weights without recording them.\n"
                    + "Use env.set_record_weights(True) (costs performance).")
            return

        vertical_size = 1.0 + 4.0 * (2)
        horizontal_size = 1.0 + 4.0 * (self.weno_order)
        fig, axes = plt.subplots(2, self.weno_order, sharex=True, sharey=True, figsize=(horizontal_size, vertical_size))

        # These arrays are time X 2 X (nx+1) X order
        learned_weights = np.array(self._all_learned_weights)
        weno_weights = np.array(self._all_weno_weights)

        if location is not None:
            learned_weights = learned_weights[:,:,location,:].transpose((1, 2, 0))
            weno_weights = weno_weights[:,:,location,:].transpose((1, 2, 0))
            fig.suptitle("weights at interface " + str(location) + " - 1/2")
        else:
            if timestep is None:
                timestep = len(self._all_learned_weights) - 1
            learned_weights = learned_weights[timestep, :, :, :].transpose((0, 2, 1))
            weno_weights = weno_weights[timestep, :, :, :].transpose((0, 2, 1))
            fig.suptitle("weights at step " + str(timestep))

        color_dim = np.arange(learned_weights.shape[2])
        offsets = np.arange(self.weno_order) - int((self.weno_order - 1)/ 2)
        for row, sign in zip((0, 1), ("+", "-")):
            for col, offset in zip(range(self.weno_order), offsets):  # TODO: figure out offset with order!=3
                label = "f^" + sign + "[" + str(offset) + "]"

                axes[row, col].set_title(label)
                if col == 0:
                    axes[row, col].set_ylabel("WENO")
                if row == 1:
                    axes[row, col].set_xlabel("RL")
                paths = axes[row, col].scatter(x=learned_weights[row, col, :], y=weno_weights[row, col, :], c=color_dim, cmap='viridis')
                axes[row, col].set(aspect='equal')
        cbar = fig.colorbar(paths, ax=axes.ravel().tolist())
        if location is not None:
            cbar.set_label("timestep")
        else:
            cbar.set_label("location")

        log_dir = logger.get_dir()
        filename = 'burgers_weights_comparison_'
        if location is not None:
            filename += 'i_' + str(location)
        else:
            filename += 'step_' + str(timestep)
        filename += '.png'

        filename = os.path.join(log_dir, filename)
        plt.savefig(filename)
        print('Saved plot to ' + filename + '.')

        plt.close(fig)

    def close(self):
        # Delete references for easier garbage collection.
        self.grid = None

    def seed(self):
        # The official Env class has this as part of its interface, but I don't think we need it. Better to set the seed at the experiment level then the environment level
        pass
