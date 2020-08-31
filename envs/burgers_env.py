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
from envs.solutions import PreciseWENOSolution, SmoothSineSolution, SmoothRareSolution, AccelShockSolution
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
        self.weno_order = weno_order
        self.grid = Grid1d(xmin=xmin, xmax=xmax, nx=nx, ng=self.ng, boundary=boundary, init_type=init_type)
        
        if srca > 0.0:
            self.source = RandomSource(grid=self.grid, amplitude=srca)
        else:
            self.source = None

        self.analytical = analytical
        if analytical:
            if init_type == "smooth_sine":
                self.solution = SmoothSineSolution(xmin=xmin, xmax=xmax, nx=nx, ng=self.ng)
            elif init_type == "smooth_rare":
                self.solution = SmoothRareSolution(xmin=xmin, xmax=xmax, nx=nx, ng=self.ng)
            elif init_type == "accelshock":
                self.solution = AccelShockSolution(xmin=xmin, xmax=xmax, nx=nx, ng=self.ng)
            else:
                raise Exception("No analytical solution available for \"{}\" type initial conditions.".format(init_type))
        else:
            if precise_weno_order is None:
                precise_weno_order = weno_order
            self.precise_weno_order = precise_weno_order
            self.precise_nx = nx * precise_scale
            self.solution = PreciseWENOSolution(xmin=xmin, xmax=xmax, nx=nx, ng=self.ng,
                                                precise_scale=precise_scale, precise_order=precise_weno_order,
                                                boundary=boundary, init_type=init_type, flux_function=flux, source=self.source,
                                                eps=eps)

        if self.analytical or self.precise_weno_order != self.weno_order or self.precise_nx != self.nx:
            self.weno_solution = PreciseWENOSolution(xmin=xmin, xmax=xmax, nx=nx, ng=self.ng,
                                                     precise_scale=1, precise_order=weno_order,
                                                     boundary=boundary, init_type=init_type, flux_function=flux,
                                                     source=self.source, eps=eps)
        else:
            self.weno_solution = None

        self.t = 0.0  # simulation time
        self.fixed_step = fixed_step
        self.C = C  # CFL number
        self.eps = eps
        self.episode_length = episode_length
        self.steps = 0
        self.action_history = []

        self._state_axes = None
        self._action_axes = None

    def step(self):
        raise NotImplementedError()

    def reset(self):
        raise NotImplementedError()

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
        if mode is None:
            return
        if "file" in mode:
            self.plot_state(**kwargs)
        if "action" in mode:
            self.plot_action(**kwargs)

    def plot_state(self, suffix=None, title=None, fixed_axes=False, no_x_borders=False, show_ghost=True):
        fig = plt.figure()

        full_x = self.grid.x
        real_x = full_x[self.ng:-self.ng]

        full_true = self.solution.get_full()
        real_true = full_true[self.ng:-self.ng]

        if self.weno_solution is not None:
            full_weno = self.weno_solution.get_full()
            real_weno = full_weno[self.ng:-self.ng]
            weno_color = "tab:blue"
            weno_ghost_color = "#75bdf0"
            true_color = "tab:pink"
            true_ghost_color = "#f7e4ed"
        else:
            true_color = "tab:blue"
            true_ghost_color = "#75bdf0"

        full_agent = self.grid.u
        real_agent = full_agent[self.ng:-self.ng]
        agent_color = "tab:orange"
        agent_ghost_color =  "#ffad66"

        # ghost green: "#94d194"

        # The ghost arrays slice off one real point so the line connects to the real points.
        # Leave off labels for these lines so they don't show up in the legend.
        num_ghost_points = self.ng + 1
        if show_ghost:
            ghost_x_left = full_x[:num_ghost_points]
            ghost_x_right = full_x[-num_ghost_points:]

            ghost_true_left = full_true[:num_ghost_points]
            ghost_true_right = full_true[-num_ghost_points:]
            plt.plot(ghost_x_left, ghost_true_left, ls='-', color=true_ghost_color)
            plt.plot(ghost_x_right, ghost_true_right, ls='-', color=true_ghost_color)

            if self.weno_solution is not None:
                ghost_weno_left = full_weno[:num_ghost_points]
                ghost_weno_right = full_weno[-num_ghost_points:]
                plt.plot(ghost_x_left, ghost_weno_left, ls='-', color=weno_ghost_color)
                plt.plot(ghost_x_right, ghost_weno_right, ls='-', color=weno_ghost_color)

            ghost_agent_left = full_agent[:num_ghost_points]
            ghost_agent_right = full_agent[-num_ghost_points:]
            plt.plot(ghost_x_left, ghost_agent_left, ls='-', color=agent_ghost_color)
            plt.plot(ghost_x_right, ghost_agent_right, ls='-', color=agent_ghost_color)

        if self.analytical:
            true_label = "Analytical"
        else:
            true_label = "WENO (order = {}, res = {})".format(self.precise_weno_order, self.precise_nx)
        plt.plot(real_x, real_true, ls='-', color=true_color, label=true_label)

        if self.weno_solution is not None:
            weno_label = "WENO (order = {}, res = {})".format(self.weno_order, self.nx)
            plt.plot(real_x, real_weno, ls='-', color=weno_color, label=weno_label)

        agent_label = "RL"
        plt.plot(real_x, real_agent, ls='-', color=agent_color, label=agent_label)

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
            if self._state_axes is None:
                self._state_axes = (ax.get_xlim(), ax.get_ylim())
            else:
                xlim, ylim = self._state_axes
                ax.set_xlim(xlim)
                ax.set_ylim(ylim)

        log_dir = logger.get_dir()
        if suffix is None:
            # Find number of digits in last step so files are sorted correctly.
            # E.g., use 001 instead of 1.
            step_precision = int(np.ceil(np.log(self.episode_length) / np.log(10)))
            suffix = ("_step{:0" + str(step_precision) + "}").format(self.steps)
        filename = 'burgers' + suffix + '.png'
        filename = os.path.join(log_dir, filename)
        plt.savefig(filename)
        print('Saved plot to ' + filename + '.')

        plt.close(fig)

    def plot_action(self, timestep=None, location=None, suffix=None, title=None, fixed_axes=False, no_x_borders=False, **kwargs):
        """
        Plot actions at either a timestep or a specific location.

        Either the timestep parameter or the location parameter can be specified, but not both.
        By default, the most recent timestep is used.

        This requires actions to be recorded in self.action_history in the subclasses step function.

        Parameters
        ----------
        timestep : int
            Timestep at which to plot actions. By default, use the most recent timestep.
        location : int
            Index of location at which to plot actions.
        suffix : string
            The plot will be saved to burgers_action_{suffix}.png. By default, the timestep/location
            is used for the suffix.
        title : string
            Title for the plot. By default, the title is based on the timestep/location.
        fixed_axes : bool
            If true, use the same axis limits on every plot. Useful for animation.
        no_x_border : bool
            If true, trim the plot to exactly the extent of the x coordinates. Useful for animation.
      
        """

        assert (timestep is None or location is None), "Can't plot action at both a timestep and a location."

        action_dimensions = np.prod(list(self.action_space.shape)[1:])

        vertical_size = 5.0
        horizontal_size = 0.5 + 3.0 * action_dimensions
        fig, axes = plt.subplots(1, action_dimensions, sharex=True, sharey=True, figsize=(horizontal_size, vertical_size))

        action_history = np.array(self.action_history)

        if self.solution.is_recording_actions():
            weno_action_history = np.array(self.solution.get_action_history())
            assert(action_history.shape == weno_action_history.shape)
        elif self.weno_solution is not None and self.weno_solution.is_recording_actions():
            weno_action_history = np.array(self.weno_solution.get_action_history())
            assert(action_history.shape == weno_action_history.shape)
        else:
            weno_action_history = None

        new_shape = (action_history.shape[0], action_history.shape[1], action_dimensions)
        action_history = action_history.reshape(new_shape)
        if weno_action_history is not None:
            weno_action_history = weno_action_history.reshape(new_shape)

        # If plotting actions at a timestep, need to transpose location to the last dimension.
        # If plotting actions at a location, need to transpose time to the last dimension.
        if location is not None:
            action_history = action_history[:,location,:].transpose()
            if weno_action_history is not None:
                weno_action_history = weno_action_history[:, location, :].transpose()

            actual_location = self.grid.x[location]
            if title is None:
                fig.suptitle("actions at x = {:.4} (i = {} - 1/2)".format(actual_location, location))
            else:
                fig.suptitle(title)
        else:
            if timestep is None:
                timestep = len(action_history) - 1
            action_history = action_history[timestep, :, :].transpose()
            if weno_action_history is not None:
                weno_action_history = weno_action_history[timestep, :, :].transpose()

            if title is None:
                if self.C is None:
                    actual_time = timestep * self.fixed_step
                    fig.suptitle("actions at t = {:.4} (step {})".format(actual_time, timestep))
                else:
                    # TODO get time with variable timesteps.
                    fig.suptitle("actions at step {}".format(actual_time, timestep))
            else:
                fig.suptitle(title)

        real_x = self.grid.inter_x[self.ng:-(self.ng-1)]

        agent_color = "tab:orange"
        weno_color = "tab:blue"
        for dim in range(action_dimensions):
            ax = axes[dim]

            if weno_action_history is not None:
                ax.plot(real_x, weno_action_history[dim, :], c=weno_color, linestyle='-', label="WENO")
            ax.plot(real_x, action_history[dim, :], c=agent_color, linestyle='-', label="RL")

            if no_x_borders:
                ax.set_xmargin(0.0)

            if fixed_axes:
               if self._action_axes is None:
                   self._action_axes = (ax.get_xlim(), ax.get_ylim())
               else:
                   xlim, ylim = self._action_axes
                   ax.set_xlim(xlim)
                   ax.set_ylim(ylim)

        plt.legend()

        log_dir = logger.get_dir()
        if suffix is None:
            if location is not None:
                location_precision = int(np.ceil(np.log(self.nx) / np.log(10)))
                suffix = ("_step{:0" + str(location_precision) + "}").format(location)
            else:
                step_precision = int(np.ceil(np.log(self.episode_length) / np.log(10)))
                suffix = ("_step{:0" + str(step_precision) + "}").format(timestep)
        filename = 'burgers_action' + suffix + '.png'

        filename = os.path.join(log_dir, filename)
        plt.savefig(filename)
        print('Saved plot to ' + filename + '.')

        plt.close(fig)

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

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.action_space = SoftmaxBox(low=0.0, high=1.0, 
                                       shape=(self.grid.real_length() + 1, 2, self.weno_order),
                                       dtype=np.float64)
        self.observation_space = spaces.Box(low=-1e7, high=1e7,
                                            shape=(self.grid.real_length() + 1, 2, 2 * self.weno_order - 1),
                                            dtype=np.float64)

        if self.weno_solution is not None:
            self.weno_solution.set_record_actions("weno")
        else:
            self.solution.set_record_actions("weno")

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
        if self.weno_solution is not None:
            self.weno_solution.reset(**self.grid.init_params)

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

        self.action_history.append(action)

        state = self.current_state

        fp_state = state[:, 0, :]
        fm_state = state[:, 1, :]

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

        if self.weno_solution is not None:
            self.weno_solution.update(dt, self.t)

        if np.isnan(state).any():
            raise Exception("NaN detected in state.")
        if np.isnan(reward).any():
            raise Exception("NaN detected in reward.")

        return state, reward, done, {}


class SplitFluxBurgersEnv(AbstractBurgersEnv):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.action_space = SoftmaxBox(low=-np.inf, high=np.inf,
                                       shape=(self.grid.real_length() + 1, 2, 2 * self.weno_order - 1),
                                       dtype=np.float64)
        self.observation_space = spaces.Box(low=-1e7, high=1e7,
                                            shape=(self.grid.real_length() + 1, 2, 2 * self.weno_order - 1),
                                            dtype=np.float64)
        if self.weno_solution is None:
            self.solution.set_record_actions("coef")
        else:
            self.weno_solution.set_record_actions("coef")

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
        if self.weno_solution is not None:
            self.weno_solution.reset(**self.grid.init_params)

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

        self.action_history.append(action)

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

        if self.weno_solution is not None:
            self.weno_solution.update(dt, self.t)

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
        if self.weno_solution is not None:
            self.weno_solution.reset(**self.grid.init_params)

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

        self.action_history.append(action)

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

        if self.weno_solution is not None:
            self.weno_solution.update(dt, self.t)

        if np.isnan(state).any():
            raise Exception("NaN detected in state.")
        if np.isnan(reward).any():
            raise Exception("NaN detected in reward.")

        return state, reward, done, {}

