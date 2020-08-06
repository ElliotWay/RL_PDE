import os
import sys
from datetime import datetime

import gym
import matplotlib.pyplot as plt
import numpy as np
from gym import spaces
from stable_baselines import logger

from envs.grid import Grid1d
from envs.source import RandomSource
from envs.solutions import PreciseWENOSolution, SmoothSineSolution, SmoothRareSolution
import envs.weno_coefficients as weno_coefficients
from util.softmax_box import SoftmaxBox
from util.misc import create_stencil_indexes

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
      stencils for each location, shape is [grid_width+1, stencil_width].
  
    Returns
    -------
    Return a batch of stencils of shape [grid_width+1, number of sub-stencils (order)].
  
    """

    a_mat = weno_coefficients.a_all[order]

    # These weights are "backwards" in the original formulation.
    # This is easier in the original formulation because we can add the k for our kth stencil to the index,
    # then subtract by a variable amount to get each value, but there's no need to do that here, and flipping
    # it back around makes the expression simpler.
    a_mat = np.flip(a_mat, axis=-1)

    stencil_indexes = create_stencil_indexes(stencil_size=order, num_stencils=order)

    stencils = np.sum(a_mat * q_batch[:, stencil_indexes], axis=-1)

    return stencils

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

def flux(q):
    return 0.5 * q ** 2

class AbstractBurgersEnv(gym.Env):
    """
    An arbitrary environment for the 1D Burgers' equation.

    This abstract class handles preparation of the state for an agent,
    the simulation of the baseline state using established methods,
    and plotting of the current state compared to the baseline state.
    How the agent sees the state, and how the agent's actions are applied to
    the state are left to subclasses.
    Namely, subclasses still need to implement step and reset, and override
    __init__ to declare the action and observation spaces.
    Subclasses should make use of self.grid, the Grid1d object that contains
    the state, and self.solution, the SolutionBase object that contains the
    baseline state.
    """
    metadata = {'render.modes': ['file']}

    def __init__(self,
            xmin=0.0, xmax=1.0, nx=128, boundary=None, init_type="smooth_sine",
            fixed_step=0.0005, C=0.5,
            weno_order=3, eps=0.0, srca=0.0, episode_length=300,
            analytical=False, precise_weno_order=None, precise_scale=1):

        self.ng = weno_order+1
        self.nx = nx
        self.grid = Grid1d(xmin=xmin, xmax=xmax, nx=nx, ng=self.ng, boundary=boundary, init_type=init_type)
        
        if srca > 0.0:
            self.source = RandomSource(nx=nx, ng=self.ng, xmin=xmin, xmax=xmax, amplitude=srca)
        else:
            self.source = None

        self.analytical = analytical
        if analytical:
            if init_type == "smooth_sine":
                self.solution = SmoothSineSolution(xmin=xmin, xmax=xmax, nx=nx, ng=self.ng)
            elif init_type == "smooth_rare":
                self.solution = SmoothRareSolution(xmin=xmin, xmax=xmax, nx=nx, ng=self.ng)
            else:
                raise Exception("No analytical solution available for \"{}\" type initial conditions.".format(init_type))
        else:
            if precise_weno_order is None:
                precise_weno_order = weno_order
            self.solution = PreciseWENOSolution(xmin=xmin, xmax=xmax, nx=nx, ng=self.ng,
                                                precise_scale=precise_scale, precise_order=precise_weno_order,
                                                boundary=boundary, init_type=init_type, flux_function=flux, source=self.source,
                                                eps=eps)

        self.t = 0.0  # simulation time
        self.fixed_step = fixed_step
        self.C = C  # CFL number
        self.weno_order = weno_order
        self.eps = eps
        self.episode_length = episode_length
        self.steps = 0

        self._solution_axes = None

    def burgers_flux(self, q):
        # This is the only thing unique to Burgers, could we make this a general class with this as a parameter?
        return 0.5 * q ** 2

    def lap(self):
        """
        Returns the Laplacian of g.u.

        This calculation relies on ghost cells, so make sure they have been filled before calling this.
        """
        # TODO - Where does this function belong? Maybe in Grid1d. Figure it out and move it there.

        gr = self.grid
        u = gr.u

        lapu = gr.scratch_array()

        ib = gr.ilo - 1
        ie = gr.ihi + 1

        lapu[ib:ie + 1] = (u[ib - 1:ie] - 2.0 * u[ib:ie + 1] + u[ib + 1:ie + 2]) / gr.dx ** 2

        return lapu

    def timestep(self):
        if self.C is None:  # return a constant time step
            return self.fixed_step
        else:
            return self.C * self.grid.dx / max(abs(self.grid.u[self.grid.ilo:
                                                          self.grid.ihi + 1]))

    def render(self, mode='file', **kwargs):
        if mode == "file":
            self.save_plot(**kwargs)
        else:
            raise Exception("Render mode: \"" + str(mode) + "\" not currently implemented.")

    def save_plot(self, suffix=None, title=None, fixed_axes=False, no_x_borders=False, show_ghost=True):
        fig = plt.figure()

        full_x = self.grid.x
        real_x = full_x[self.ng:-self.ng]

        full_actual = self.solution.get_full()
        real_actual = full_actual[self.ng:-self.ng]

        full_learned = self.grid.u
        real_learned = full_learned[self.ng:-self.ng]

        # The ghost arrays slice off one real point so the line connects to the real points.
        # Leave off labels for these lines so they don't show up in the legend.
        num_ghost_points = self.ng + 1
        if show_ghost:
            ghost_x_left = full_x[:num_ghost_points]
            ghost_x_right = full_x[-num_ghost_points:]
            ghost_actual_left = full_actual[:num_ghost_points]
            ghost_actual_right = full_actual[-num_ghost_points:]
            ghost_blue = "#80b0ff"
            plt.plot(ghost_x_left, ghost_actual_left, ls='-', color=ghost_blue)
            plt.plot(ghost_x_right, ghost_actual_right, ls='-', color=ghost_blue)
            ghost_learned_left = full_learned[:num_ghost_points]
            ghost_learned_right = full_learned[-num_ghost_points:]
            ghost_black = "#808080"  # ie grey
            plt.plot(ghost_x_left, ghost_learned_left, ls='-', color=ghost_black)
            plt.plot(ghost_x_right, ghost_learned_right, ls='-', color=ghost_black)

        actual_label = "WENO"
        if self.analytical:
            actual_label = "Analytical"
        plt.plot(real_x, real_actual, ls='-', color='b', label=actual_label)

        plt.plot(real_x, real_learned, ls='-', color='k', label="RL")
        plt.legend()

        ax = plt.gca()

        # Recalculate automatic axis scaling.
        #ax.relim()
        #ax.autoscale_view()

        if title is None:
            title = "t = {:.3f}s".format(self.t)

        ax.set_title(title)

        if no_x_borders:
            ax.set_xmargin(0.0)

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

    def step(self):
        raise NotImplementedError()

    def reset(self):
        raise NotImplementedError()

    def calculate_reward(self):
        """ Optional reward calculation based on the error between grid and solution. """

        done = False

        # Error-based reward.
        reward = 0.0
        error = self.solution.get_full() - self.grid.get_full()

        # Clip tiny errors.
        #error[error < 0.001] = 0
        # Enhance extreme errors.
        #error[error > 0.1] *= 10

        # error = error[self.ng:-self.ng]
        # These give reward based on error in cell right of interface, so missing reward for rightmost interface.
        #reward = np.max(np.abs(error))
        #reward = (error) ** 2

        # Reward as function of the errors in the stencil.
        # max error across stencil
        stencil_indexes = create_stencil_indexes(
                stencil_size=(self.weno_order * 2 - 1),
                num_stencils=(self.nx + 1),
                offset=(self.ng - self.weno_order))
        error_stencils = error[stencil_indexes]
        reward = np.amax(np.abs(error_stencils), axis=-1)
        #reward = np.sqrt(np.sum(error_stencils**2, axis=-1))

        # Squash error.
        reward = -np.arctan(reward)
        max_penalty = np.pi / 2
        #reward = -error
        #max_penalty = 1e7


        # Conservation-based reward.
        # reward = -np.log(np.sum(rhs[g.ilo:g.ihi+1]))

        # Give a penalty and end the episode if we're way off.
        #if np.max(state) > 1e7 or np.isnan(np.max(state)): state possibly made more sense here
        if np.max(error) > 1e7 or np.isnan(np.max(error)):
            reward -= max_penalty * (self.episode_length - self.steps)
            done = True

        return reward, done

    def close(self):
        # Delete references for easier garbage collection.
        self.grid = None
        self.solution = None

    def seed(self):
        # The official Env class has this as part of its interface, but I don't think we need it. Better to set the numpy seed at the experiment level.
        pass


class WENOBurgersEnv(AbstractBurgersEnv):

    def __init__(self, *args, record_weights=False, **kwargs):
        super().__init__(*args, **kwargs)
        self.record_weights = record_weights

        self.action_space = SoftmaxBox(low=0.0, high=1.0, 
                                       shape=(self.grid.real_length() + 1, 2, self.weno_order),
                                       dtype=np.float64)
        self.observation_space = spaces.Box(low=-1e7, high=1e7,
                                            shape=(self.grid.real_length() + 1, 2, 2 * self.weno_order - 1),
                                            dtype=np.float64)

    def set_record_weights(self, record_weights):
        self.record_weights = record_weights

    def prep_state(self):
        """
        Return state at current time step. Returns fpr and fml vector slices.
  
        Returns
        -------
        state: np array.
            State vector to be sent to the policy. 
            Size: (grid_size + 1) BY 2 (one for fp and one for fm) BY stencil_size
            Example: order =3, grid = 128
  
        """
        # get the solution data
        g = self.grid

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

        # Stack fp and fm on axis 1 so grid position is still axis 0.
        state = np.stack([fp_stencils, fm_stencils], axis=1)

        # save this state for convenience
        self.current_state = state

        return state

    def reset(self):
        """
        Reset the environment.

        Returns
        -------
        Initial state.

        """
        self.grid.reset()
        if self.source is not None:
            self.source.reset()
        self.solution.reset(**self.grid.init_params)

        self.t = 0.0
        self.steps = 0

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
          size: (grid-size + 1) X 2 (fp, fm) X order

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

        if np.isnan(action).any():
            raise Exception("NaN detected in action.")

        state = self.current_state

        fp_state = state[:, 0, :]
        fm_state = state[:, 1, :]

        # Record weights if that mode is enabled.
        if self.record_weights:
            self._all_learned_weights.append(action)
            weno_weights_fp = weno_weights_batch(self.weno_order, fp_state)
            weno_weights_fm = weno_weights_batch(self.weno_order, fm_state)
            weno_weights = np.array([weno_weights_fp, weno_weights_fm])
            self._all_weno_weights.append(weno_weights)

        done = False
        g = self.grid

        # Store the data at the start of the time step
        u_copy = g.get_real()

        # Should probably find a way to pass this to agent if dt varies.
        dt = self.timestep()

        fp_stencils = weno_i_stencils_batch(self.weno_order, fp_state)
        fm_stencils = weno_i_stencils_batch(self.weno_order, fm_state)

        fp_action = action[:, 0, :]
        fm_action = action[:, 1, :]

        fpr = np.sum(fp_action * fp_stencils, axis=-1)
        fml = np.sum(fm_action * fm_stencils, axis=-1)

        flux = fml + fpr

        if self.eps > 0.0:
            R = self.eps * self.lap()
            rhs = (flux[:-1] - flux[1:]) / g.dx + R[g.ilo:g.ihi+1]
        else:
            rhs = (flux[:-1] - flux[1:]) / g.dx

        if self.source is not None:
            self.source.update(dt, self.t + dt)
            rhs += self.source.get_real()

        u_copy += dt * rhs
        self.grid.update(u_copy)

        # update the solution time
        self.t += dt

        # Calculate new RL state.
        state = self.prep_state()

        self.steps += 1
        if self.steps >= self.episode_length:
            done = True

        self.solution.update(dt, self.t)
        reward, force_done = self.calculate_reward()

        done = done or force_done

        if np.isnan(state).any():
            raise Exception("NaN detected in state.")
        if np.isnan(reward).any():
            raise Exception("NaN detected in reward.")

        return state, reward, done, {}

    def plot_weights(self, timestep=None, location=None):
        assert(len(self._all_learned_weights) == len(self._all_weno_weights))
        if len(self._all_learned_weights) == 0:
            print("Can't plot weights without recording them.\n"
                    + "Use env.set_record_weights(True) (costs performance).")
            return

        vertical_size = 1.0 + 4.0 * (2)
        horizontal_size = 1.0 + 4.0 * (self.weno_order)
        fig, axes = plt.subplots(2, self.weno_order, sharex=True, sharey=True, figsize=(horizontal_size, vertical_size))

        # These arrays are time X (nx+1) X 2 X order
        learned_weights = np.array(self._all_learned_weights)
        weno_weights = np.array(self._all_weno_weights)

        if location is not None:
            learned_weights = learned_weights[:,location,:,:].transpose((1, 2, 0))
            weno_weights = weno_weights[:,location,:,:].transpose((1, 2, 0))
            fig.suptitle("weights at interface " + str(location) + " - 1/2")
        else:
            if timestep is None:
                timestep = len(self._all_learned_weights) - 1
            learned_weights = learned_weights[timestep, :, :, :].transpose((1, 2, 0))
            weno_weights = weno_weights[timestep, :, :, :].transpose((1, 2, 0))
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

class SplitFluxBurgersEnv(AbstractBurgersEnv):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.action_space = SoftmaxBox(low=-np.inf, high=np.inf,
                                       shape=(self.grid.real_length() + 1, 2, 2 * self.weno_order - 1),
                                       dtype=np.float64)
        self.observation_space = spaces.Box(low=-1e7, high=1e7,
                                            shape=(self.grid.real_length() + 1, 2, 2 * self.weno_order - 1),
                                            dtype=np.float64)

    def prep_state(self):
        """
        Return state at current time step. Returns fpr and fml vector slices.
  
        Returns
        -------
        state: np array.
            State vector to be sent to the policy. 
            Size: (grid_size + 1) BY 2 (one for fp and one for fm) BY stencil_size
            Example: order =3, grid = 128
  
        """
        # get the solution data
        g = self.grid

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

        # Stack fp and fm on axis 1 so grid position is still axis 0.
        state = np.stack([fp_stencils, fm_stencils], axis=1)

        # save this state for convenience
        self.current_state = state

        return state

    def reset(self):
        """
        Reset the environment.

        Returns
        -------
        Initial state.

        """
        self.grid.reset()
        if self.source is not None:
            self.source.reset()
        self.solution.reset(**self.grid.init_params)

        self.t = 0.0
        self.steps = 0

        return self.prep_state()

    def step(self, action):
        """
        Perform a single time step.

        Parameters
        ----------
        action : np array
          Weights for construction the flux.
          size: (grid-size + 1) X 2 (fp, fm) X (2 * order - 1)

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

        if np.isnan(action).any():
            raise Exception("NaN detected in action.")

        state = self.current_state

        fp_state = state[:, 0, :]
        fm_state = state[:, 1, :]

        done = False
        g = self.grid

        # Store the data at the start of the time step
        u_copy = g.get_real()

        # Should probably find a way to pass this to agent if dt varies.
        dt = self.timestep()

        fp_action = action[:, 0, :]
        fm_action = action[:, 1, :]

        fpr = np.sum(fp_action * fp_state, axis=-1)
        fml = np.sum(fm_action * fm_state, axis=-1)

        flux = fml + fpr

        if self.eps > 0.0:
            R = self.eps * self.lap()
            rhs = (flux[:-1] - flux[1:]) / g.dx + R[g.ilo:g.ihi+1]
        else:
            rhs = (flux[:-1] - flux[1:]) / g.dx

        if self.source is not None:
            self.source.update(dt, self.t + dt)
            rhs += self.source.get_real()

        u_copy += dt * rhs
        self.grid.update(u_copy)

        # update the solution time
        self.t += dt

        # Calculate new RL state.
        state = self.prep_state()

        self.steps += 1
        if self.steps >= self.episode_length:
            done = True

        self.solution.update(dt, self.t)

        reward, force_done = self.calculate_reward()
        done = done or force_done

        if np.isnan(state).any():
            raise Exception("NaN detected in state.")
        if np.isnan(reward).any():
            raise Exception("NaN detected in reward.")

        return state, reward, done, {}


class FluxBurgersEnv(AbstractBurgersEnv):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.action_space = SoftmaxBox(low=-np.inf, high=np.inf,
                                       shape=(self.grid.real_length() + 1, 2 * self.weno_order),
                                       dtype=np.float64)
        self.observation_space = spaces.Box(low=-1e7, high=1e7,
                                            shape=(self.grid.real_length() + 1, 2 * self.weno_order),
                                            dtype=np.float64)

    def prep_state(self):
        """
        Return state at current time step. State is flux.
  
        Returns
        -------
        state: np array.
            State vector to be sent to the policy. 
            Size: (grid_size + 1) BY stencil_size (2 * order)
  
        """
        # get the solution data
        g = self.grid

        # compute flux at each point
        f = self.burgers_flux(g.u)

        # (Do not split flux for this version.)

        # stencil_size is 1 more than in other versions. This is because we
        # need to include the stencils for both + and -.
        stencil_indexes = create_stencil_indexes(stencil_size=self.weno_order * 2,
                                                 num_stencils=g.real_length() + 1,
                                                 offset=g.ng - self.weno_order
                                                 )

        stencils = f[stencil_indexes]
        state = stencils

        # save this state for convenience
        self.current_state = state

        return state

    def reset(self):
        """
        Reset the environment.

        Returns
        -------
        Initial state.

        """
        self.grid.reset()
        if self.source is not None:
            self.source.reset()
        self.solution.reset(**self.grid.init_params)

        self.t = 0.0
        self.steps = 0

        return self.prep_state()

    def step(self, action):
        """
        Perform a single time step.

        Parameters
        ----------
        action : np array
          Weights for construction the flux.
          size: (grid-size + 1) X 2 (fp, fm) X (2 * order - 1)

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

        if np.isnan(action).any():
            raise Exception("NaN detected in action.")

        state = self.current_state

        done = False
        g = self.grid

        # Store the data at the start of the time step
        u_copy = g.get_real()

        # Should probably find a way to pass this to agent if dt varies.
        dt = self.timestep()

        flux = np.sum(action * state, axis=-1)

        if self.eps > 0.0:
            R = self.eps * self.lap()
            rhs = (flux[:-1] - flux[1:]) / g.dx + R[g.ilo:g.ihi+1]
        else:
            rhs = (flux[:-1] - flux[1:]) / g.dx

        if self.source is not None:
            self.source.update(dt, self.t + dt)
            rhs += self.source.get_real()

        u_copy += dt * rhs
        self.grid.update(u_copy)

        # update the solution time
        self.t += dt

        # Calculate new RL state.
        state = self.prep_state()

        self.steps += 1
        if self.steps >= self.episode_length:
            done = True

        self.solution.update(dt, self.t)

        reward, force_done = self.calculate_reward()
        done = done or force_done

        if np.isnan(state).any():
            raise Exception("NaN detected in state.")
        if np.isnan(reward).any():
            raise Exception("NaN detected in reward.")

        return state, reward, done, {}

