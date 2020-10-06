import os
import sys
from datetime import datetime

import gym
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
from gym import spaces
from stable_baselines import logger

from envs.grid import Grid1d
from envs.source import RandomSource
from envs.solutions import PreciseWENOSolution, AnalyticalSolution, MemoizedSolution
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
            analytical=False, precise_weno_order=None, precise_scale=1,
            reward_adjustment=1000,
            memoize=False,
            test=False):

        self.test = test

        self.ng = weno_order+1
        self.nx = nx
        self.weno_order = weno_order
        self.grid = Grid1d(xmin=xmin, xmax=xmax, nx=nx, ng=self.ng,
                           boundary=boundary, init_type=init_type,
                           deterministic_init=self.test)
        
        if srca > 0.0:
            self.source = RandomSource(grid=self.grid, amplitude=srca)
        else:
            self.source = None

        self.analytical = analytical
        if analytical:
            self.solution = AnalyticalSolution(xmin=xmin, xmax=xmax, nx=nx, ng=self.ng, init_type=init_type)
        else:
            if precise_weno_order is None:
                precise_weno_order = weno_order
            self.precise_weno_order = precise_weno_order
            self.precise_nx = nx * precise_scale
            self.solution = PreciseWENOSolution(
                    xmin=xmin, xmax=xmax, nx=nx, ng=self.ng,
                    precise_scale=precise_scale, precise_order=precise_weno_order,
                    boundary=boundary, init_type=init_type, flux_function=flux, source=self.source,
                    eps=eps)
        if memoize:
            self.solution = MemoizedSolution(self.solution)

        # Disable this for now. The original idea was that you could compare both WENO and the learned RL solution to the analytical solution.
        if False: #self.analytical or self.precise_weno_order != self.weno_order or self.precise_nx != self.nx:
            self.weno_solution = PreciseWENOSolution(xmin=xmin, xmax=xmax, nx=nx, ng=self.ng,
                                                     precise_scale=1, precise_order=weno_order,
                                                     boundary=boundary, init_type=init_type, flux_function=flux,
                                                     source=self.source, eps=eps)
        else:
            self.weno_solution = None

        self.fixed_step = fixed_step
        self.C = C  # CFL number
        self.eps = eps
        self.episode_length = episode_length
        self.reward_adjustment = reward_adjustment

        self._step_precision = int(np.ceil(np.log(self.episode_length) / np.log(10)))
        self._cell_index_precision = int(np.ceil(np.log(self.nx) / np.log(10)))

        self.t = 0.0
        self.steps = 0
        self.action_history = []
        self.state_history = []

        self._state_axes = None
        self._action_axes = None
        self._action_labels = None

        if self.weno_solution is not None:
            self.weno_color = "tab:blue"
            self.weno_ghost_color = "#75bdf0"
            self.true_color = "tab:pink"
            self.true_ghost_color = "#f7e4ed"
        else:
            self.true_color = "tab:blue"
            self.true_ghost_color = "#75bdf0"
        self.agent_color = "tab:orange"
        self.agent_ghost_color =  "#ffad66"
        # ghost green: "#94d194"

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
            return self.plot_state(**kwargs)
        if "action" in mode:
            return self.plot_action(**kwargs)

    def get_state(self, timestep=None, location=None, full=True):
        assert timestep is None or location is None

        if timestep is None and location is None:
            state = self.grid.get_full() if full else self.grid.get_real()
        elif timestep is not None:
            state = self.state_history[timestep, :]
            if not full:
                state = state[self.ng:-self.ng]
        else:
            state = self.state_history[:, location]
        return state

    def get_solution_state(self, timestep=None, location=None, full=True):
        assert timestep is None or location is None

        if timestep is None and location is None:
            state = self.solution.get_full() if full else self.solution.get_real()
        elif timestep is not None:
            state = self.solution.get_state_history()[timestep, :]
            if not full:
                state = state[self.ng:-self.ng]
        else:
            state = self.solution.get_state_history()[:, location]
        return state

    def get_error(self, timestep=None, location=None, full=True):
        return (self.get_state(timestep, location, full) 
                - self.get_solution_state(timestep, location, full))

    def compute_l2_error(self, timestep=None):
        if timestep == "all":
            l2_errors = []
            for step in range(len(self.state_history)):
                l2_errors.append(self.compute_l2_error(step))
            return l2_errors

        else:
            error = self.get_error(timestep=timestep, full=False)
            l2_error = np.sqrt(self.grid.dx * np.sum(np.square(error)))
            return l2_error

    def plot_state(self, timestep=None, location=None, plot_error=False,
            suffix=None, title=None, fixed_axes=False, no_x_borders=False, show_ghost=True):
        """
        Plot environment state at either a timestep or a specific location.

        Either the timestep parameter or the location parameter can be specified, but not both.
        By default, the most recent timestep is used.

        The default only requires self.grid to be updated. Specifying a time or location requires
        the state to be recorded in self.state_history in the subclass's step function.

        Parameters
        ----------
        timestep : int
            Timestep at which to plot state. By default, use the most recent timestep.
        location : int
            Index of location at which to plot actions.
        plot_error : int
            Plot the error of the state with the solution state instead.
        suffix : string
            The plot will be saved to burgers_state_{suffix}.png (or burgers_error_{suffix}.png).
            By default, the timestep/location is used for the suffix.
        title : string
            Title for the plot. By default, the title is based on the timestep/location.
        fixed_axes : bool
            If true, use the same axis limits on every plot. Useful for animation.
        no_x_border : bool
            If true, trim the plot to exactly the extent of the x coordinates. Useful for animation.
        show_ghost : bool
            Plot the ghost cells in addition to the "real" cells. The ghost cells are plotted
            in a lighter color.
        """

        assert (timestep is None or location is None), "Can't plot state at both a timestep and a location."

        fig = plt.figure()

        error_or_state = "error" if plot_error else "state"

        if location is None and timestep is None:
            state_history = self.grid.get_full().copy()
            solution_state_history = self.solution.get_full().copy()
            if self.weno_solution is not None:
                weno_state_history = self.weno_solution.get_full().copy()
            else:
                weno_state_history = None

            if title is None:
                title = ("{} at t = {:.4f} (step {:0" + str(self._step_precision) + "d})").format(error_or_state, self.t, self.steps)
            if suffix is None:
                suffix = ("_step{:0" + str(self._step_precision) + "}").format(self.steps)
        else:
            state_history = np.array(self.state_history)
            solution_state_history = self.solution.get_state_history().copy() \
                    if self.solution.is_recording_state() else None
            weno_state_history = self.weno_solution.get_state_history().copy() \
                    if self.weno_solution is not None and self.weno_solution.is_recording_state() else None

            if location is not None:
                location = self.ng + location
                state_history = state_history[:,location]
                if solution_state_history is not None:
                    solution_state_history = solution_state_history[:, location]
                if weno_state_history is not None:
                    weno_state_history = weno_state_history[:, location]

                actual_location = self.grid.x[location]
                if title is None:
                    title = "{} at x = {:.4f} (i = {})".format(error_or_state, actual_location, location)
                if suffix is None:
                    suffix = ("_step{:0" + str(self._cell_index_precision) + "}").format(location)
            else:
                state_history = state_history[timestep, :]
                if solution_state_history is not None:
                    solution_state_history = solution_state_history[timestep, :]
                if weno_state_history is not None:
                    weno_state_history = weno_state_history[timestep, :]

                if title is None:
                    if self.C is None:
                        actual_time = timestep * self.fixed_step
                        title = ("{} at t = {:.4f} (step {:0" + str(self._step_precision) + "d})").format(error_or_state, actual_time, timestep)
                    else:
                        # TODO get time with variable timesteps?
                        title = "{} at step {:0" + str(self._step_precision) + "d}".format(error_or_state, timestep)
                if suffix is None:
                    suffix = ("_step{:0" + str(self._step_precision) + "}").format(timestep)

        if plot_error:
            if solution_state_history is None:
                raise Exception("Cannot plot error if solution state is not available.")

            state_history = np.abs(solution_state_history - state_history)
            solution_state_history = None
            weno_state_history = None

        if timestep is None:
            if show_ghost:
                # The ghost arrays slice off one real point so the line connects to the real points.
                num_ghost_points = self.ng + 1

                ghost_x_left = self.grid.x[:num_ghost_points]
                ghost_x_right = self.grid.x[-num_ghost_points:]

                plt.plot(ghost_x_left, state_history[:num_ghost_points], ls='-', color=self.agent_ghost_color)
                plt.plot(ghost_x_right, state_history[-num_ghost_points:], ls='-', color=self.agent_ghost_color)

                if solution_state_history is not None:
                    plt.plot(ghost_x_left, solution_state_history[:num_ghost_points], ls='-', color=self.true_ghost_color)
                    plt.plot(ghost_x_right, solution_state_history[-num_ghost_points:], ls='-', color=self.true_ghost_color)

                if weno_state_history is not None:
                    plt.plot(ghost_x_left, weno_state_history[:num_ghost_points], ls='-', color=self.weno_ghost_color)
                    plt.plot(ghost_x_right, weno_state_history[-num_ghost_points:], ls='-', color=self.weno_ghost_color)

            state_history = state_history[self.ng:-self.ng]
            if solution_state_history is not None:
                solution_state_history = solution_state_history[self.ng:-self.ng]
            if weno_state_history is not None:
                weno_state_history = weno_state_history[self.ng:-self.ng]

        if timestep is None:
            x_values = self.grid.x[self.ng:-self.ng]
        else:
            if self.C is None:
                x_values = self.fixed_step * np.arange(len(state_history))
            else:
                # Need to record time values with variable timesteps.
                x_values = np.arange(len(state_history))

        if self.analytical:
            true_label = "Analytical"
            weno_label = "WENO"
        else:
            if self.weno_solution is not None:
                true_label = "WENO (order={}, res={})".format(self.precise_weno_order, self.precise_nx)
                weno_label = "WENO (order={}, res={})".format(self.weno_order, self.nx)
            else:
                true_label = "WENO"
        if solution_state_history is not None:
            plt.plot(x_values, solution_state_history, ls='-', color=self.true_color, label=true_label)
        if weno_state_history is not None:
            plt.plot(x_values, weno_state_history, ls='-', color=self.weno_color, label=weno_label)

        # Plot this one last so it is on the top.
        agent_label = "RL"
        if plot_error:
            agent_label = "|error|"
        plt.plot(x_values, state_history, ls='-', color=self.agent_color, label=agent_label)

        plt.legend(loc="upper right")
        ax = plt.gca()

        ax.set_title(title)

        if no_x_borders:
            ax.set_xmargin(0.0)

        # Restrict y-axis if plotting abs error.
        # Can't have negative, cut off extreme errors.
        if plot_error:
            extreme_cutoff = 3.0
            max_not_extreme = np.max(state_history[state_history < extreme_cutoff])
            ymax = max_not_extreme*1.05 if max_not_extreme > 0.0 else 0.01
            ax.set_ylim((0.0, ymax))

        if fixed_axes:
            if self._state_axes is None:
                self._state_axes = (ax.get_xlim(), ax.get_ylim())
            else:
                xlim, ylim = self._state_axes
                ax.set_xlim(xlim)
                ax.set_ylim(ylim)

        log_dir = logger.get_dir()
        filename = "burgers_{}{}.png".format(error_or_state, suffix)
        filename = os.path.join(log_dir, filename)
        plt.savefig(filename)
        print('Saved plot to ' + filename + '.')

        plt.close(fig)
        return filename

    def plot_state_evolution(self, num_states=10, full_true=False, no_true=False, plot_error=False, suffix="", title=None):
        """
        Plot the evolution of the state over time on a single plot.
        Ghost cells are not plotted.
        Time delta is not indicated anywhere on the plot.

        Parameters
        ----------
        num_states : int
            Number of states to plot. Does not include the initial state
            but does include the final state, so num_states=10 will have 11
            lines.
        plot_error : bool
            Plot the evolution of the error of the state with the solution state
            instead. Overrides full_true and no_true.
        full_true : bool
            Set False by default, which will only plot the final state of the
            true solution. Set True to plot the state of the true solution at
            every the same number of timesteps as the RL solution. (This may
            be confusing to interpret.)
        no_true : bool
            Set False by default, which plots the true solution. Set True to
            ONLY plot the RL solution, if you don't care about the true solution.
            Also useful to plot evolution of the true solution itself.
        suffix : string
            The plot will be saved to burgers_evolution_state{suffix}.png
            (or burgers_evolution_error{suffix}.png). There is no suffix by
            default.
        title : string
            Title for the plot. There is no title by default.
        """

        fig = plt.figure()

        x_values = self.grid.x[self.ng:-self.ng]

        state_history = np.array(self.state_history)[:, self.ng:-self.ng]

        # Indexes into the state history. There are num_states+1 indices, where the first
        # is always 0m the last is always len(state_history)-1, and the rest are evenly
        # spaced between them.
        slice_indexes = (np.arange(num_states+1)*(len(state_history)-1)/num_states).astype(int)

        # Create a color sequence, in the same structure as the above slice indexes.
        # The first is always start_vec, the last is always end_vec, and the rest
        # are evenly spaced.
        def color_sequence(start_vec, end_vec, n):
            return list(zip(*[
                start + np.arange(n+1)*((end-start)/n)
                for start, end in zip(start_vec, end_vec)]))

        start_rgb = (0.9, 0.9, 0.9) #light grey

        # Plot the true solution first so it appears under the RL solution.
        if not plot_error and not no_true and self.solution.is_recording_state():
            solution_state_history = np.array(self.solution.get_state_history())[:, self.ng:-self.ng]

            assert len(state_history) == len(solution_state_history), "History mismatch."
            
            if full_true:
                true_rgb = matplotlib.colors.to_rgb(self.true_color)
                true_color_sequence = color_sequence(start_rgb, true_rgb, num_states)
                sliced_solution_history = solution_state_history[slice_indexes]
                
                for state_values, color in zip(sliced_solution_history[1:-1], true_color_sequence[1:-1]):
                    plt.plot(x_values, state_values, ls='-', linewidth=1, color=color)
                true = plt.plot(x_values, solution_state_history[-1],
                                ls='-', linewidth=1, color=true_rgb)
            else:
                true = plt.plot(x_values, solution_state_history[-1], 
                                ls='-', linewidth=4, color=self.true_color)
        else:
            true = None

        if plot_error:
            if not self.solution.is_recording_state():
                raise Exception("Cannot plot evolution of error if solution state is not available.")
            solution_state_history = np.array(self.solution.get_state_history())[:, self.ng:-self.ng]
            state_history = np.abs(solution_state_history - state_history)

        if not plot_error:
            init = plt.plot(x_values, state_history[0], ls='--', color=start_rgb)

        if full_true and not plot_error and not no_true:
            agent_rgb = matplotlib.colors.to_rgb(self.agent_color)
        else:
            # Use black if no color contrast is needed.
            agent_rgb = (0.0, 0.0, 0.0)
        agent_color_sequence = color_sequence(start_rgb, agent_rgb, num_states)
        sliced_history = state_history[slice_indexes]
        for state_values, color in zip(sliced_history[1:-1], agent_color_sequence[1:-1]):
            plt.plot(x_values, state_values, ls='-', color=color)
        agent = plt.plot(x_values, state_history[-1], ls='-', color=agent_rgb)

        ax = plt.gca()
        if plot_error:
            plots = [agent[0]]
            labels = ["|error|"]
        else:
            plots = [init[0], agent[0]]
            labels = ["init", "RL"]
            if true is not None:
                plots.append(true[0])
                if self.analytical:
                    labels.append("Analytical")
                else:
                    labels.append("WENO")
        ax.legend(plots, labels)

        if title is not None:
            ax.set_title(title)

        ax.set_xmargin(0.0)
        ax.set_xlabel('x')
        ax.set_ylabel('u')
        
        log_dir = logger.get_dir()
        error_or_state = "error" if plot_error else "state"
        filename = os.path.join(log_dir,
                "burgers_evolution_{}{}.png".format(error_or_state, suffix))
        plt.savefig(filename)
        print('Saved plot to ' + filename + '.')
        
        plt.close(fig)
        return filename

    def plot_action(self, timestep=None, location=None, suffix=None, title=None,
                    fixed_axes=False, no_x_borders=False, **kwargs):
        """
        Plot actions at either a timestep or a specific location.

        Either the timestep parameter or the location parameter can be specified, but not both.
        By default, the most recent timestep is used.

        This requires actions to be recorded in self.action_history in the subclass's step function.

        Parameters
        ----------
        timestep : int
            Timestep at which to plot actions. By default, use the most recent timestep.
        location : int
            Index of location at which to plot actions.
        suffix : string
            The plot will be saved to burgers_action{suffix}.png. By default, the timestep/location
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
            weno_color = self.true_color
            assert(action_history.shape == weno_action_history.shape)
        elif self.weno_solution is not None and self.weno_solution.is_recording_actions():
            weno_action_history = np.array(self.weno_solution.get_action_history())
            weno_color = self.weno_color
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

            actual_location = self.grid.x[location] - self.dx/2
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

        for dim in range(action_dimensions):
            ax = axes[dim]

            if weno_action_history is not None:
                ax.plot(real_x, weno_action_history[dim, :], c=weno_color, linestyle='-', label="WENO")
            ax.plot(real_x, action_history[dim, :], c=self.agent_color, linestyle='-', label="RL")

            if no_x_borders:
                ax.set_xmargin(0.0)

            if fixed_axes:
               if self._action_axes is None:
                   self._action_axes = (ax.get_xlim(), ax.get_ylim())
               else:
                   xlim, ylim = self._action_axes
                   ax.set_xlim(xlim)
                   ax.set_ylim(ylim)

            if self._action_labels is not None:
                ax.set_title(self._action_labels[dim])

        plt.legend()

        log_dir = logger.get_dir()
        if suffix is None:
            if location is not None:
                suffix = ("_step{:0" + str(self._cell_index_precision) + "}").format(location)
            else:
                suffix = ("_step{:0" + str(self._step_precision) + "}").format(timestep)
        filename = 'burgers_action' + suffix + '.png'

        filename = os.path.join(log_dir, filename)
        plt.savefig(filename)
        print('Saved plot to ' + filename + '.')

        plt.close(fig)
        return filename

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

        # Reward as average of error in two adjacent cells.
        error = np.abs(error)
        reward = (error[self.ng-1:-self.ng] + error[self.ng:-(self.ng-1)]) / 2

        # Reward as function of the errors in the stencil.
        # max error across stencil
        #stencil_indexes = create_stencil_indexes(
                #stencil_size=(self.weno_order * 2 - 1),
                #num_stencils=(self.nx + 1),
                #offset=(self.ng - self.weno_order))
        #error_stencils = error[stencil_indexes]
        #reward = np.amax(np.abs(error_stencils), axis=-1)
        #reward = np.sqrt(np.sum(error_stencils**2, axis=-1))

        # Squash error.
        #reward = -np.arctan(reward)

        # The constant controls the relative importance of small rewards compared to large rewards.
        # Towards infinity, all rewards (or penalties) are equally important.
        # Towards 0, small rewards are increasingly less important.
        # An alternative to arctan(C*x) with this property would be x^(1/C).
        reward = -np.arctan(self.reward_adjustment * reward)

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

    def evolve(self):
        """ Evolve the environment using the solution, instead of passing actions. """

        while self.steps < self.episode_length:
            
            dt = self.timestep()
            self.t += dt

            if self.source is not None:
                self.source.update(dt, self.t)

            self.solution.update(dt, self.t)

            self.grid.set(self.solution.get_real().copy())
            self.state_history.append(self.grid.get_full().copy())

            if self.solution.is_recording_actions():
                self.action_history.append(self.solution.get_action_history()[-1].copy())

            self.steps += 1

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
        
        self.solution.set_record_state(True)
        if self.weno_solution is not None:
            self.weno_solution.set_record_state(True)

        if self.weno_solution is not None:
            self.weno_solution.set_record_actions("weno")
        elif not self.analytical:
            self.solution.set_record_actions("weno")

        self.k1 = self.k2 = self.k3 = self.u_start = self.dt = None
        self.rk_state = 1

        self._action_labels = ["$w^{}_{}$".format(sign, num) for sign in ['+', '-']
                                    for num in range(1, self.weno_order+1)]

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
        self.solution.reset(self.grid.init_params)
        if self.weno_solution is not None:
            self.weno_solution.reset(self.grid.init_params)

        self.state_history = [self.grid.get_full().copy()]

        self.rk_state = 1

        self.t = 0.0
        self.steps = 0

        return self.prep_state()

    def rk_substep_weno(self, weights):

        state = self.current_state

        fp_state = state[:, 0, :]
        fm_state = state[:, 1, :]

        fp_stencils = weno_i_stencils_batch(self.weno_order, fp_state)
        fm_stencils = weno_i_stencils_batch(self.weno_order, fm_state)

        fp_weights = weights[:, 0, :]
        fm_weights = weights[:, 1, :]

        fpr = np.sum(fp_weights * fp_stencils, axis=-1)
        fml = np.sum(fm_weights * fm_stencils, axis=-1)

        flux = fml + fpr

        rhs = (flux[:-1] - flux[1:]) / self.grid.dx

        return rhs

    def rk4_step(self, action):
        """
        Perform one RK4 substep.

        You should call this function 4 times in succession.
        The 1st, 2nd, and 3rd calls will return the state in each substep.
        The 4th call will return the state after the full RK4 step.
        
        Regardless of the internal state, the action should always be based on the previously returned state.

        Returns
        -------
        state: np array
          depends on which of the 4th calls this is
        reward: np array
          0 on first 3 calls, reward for each location on the 4th call
        done: boolean
          end of episode, guaranteed to be False on first 3 calls
        """
        if np.isnan(action).any():
            raise Exception("NaN detected in action.")

        if self.rk_state == 1:
            self.u_start = np.array(self.grid.get_real())
            self.dt = self.timestep()

            self.k1 = self.dt * self.rk_substep_weno(action)
            self.grid.set(self.u_start + self.k1/2)
            state = self.prep_state()

            self.rk_state = 2
            return state, np.zeros_like(action), False
        elif self.rk_state == 2:
            self.k2 = self.dt * self.rk_substep_weno(action)
            self.grid.set(self.u_start + self.k2/2)
            state = self.prep_state()

            self.rk_state = 3
            return state, np.zeros_like(action), False
        elif self.rk_state == 3:
            self.k3 = self.dt * self.rk_substep_weno(action)
            self.grid.set(self.u_start + self.k3)
            state = self.prep_state()

            self.rk_state = 4
            return state, np.zeros_like(action), False
        else:
            assert self.rk_state == 4

            k4 = self.dt * self.rk_substep_weno(action)
            step = (self.k1 + 2*(self.k2 + self.k3) + k4) / 6

            state, reward, done = self._finish_step(step, self.dt, prev=self.u_start)

            self.rk_state = 1
            self.k1 = self.k2 = self.k3 = self.u_start = self.dt = None
            return state, reward, done


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

        # Should probably find a way to pass this to agent if dt varies.
        dt = self.timestep()

        step = dt * self.rk_substep_weno(action)

        state, reward, done = self._finish_step(step, dt)

        self.action_history.append(action)
        self.state_history.append(self.grid.get_full().copy())

        return state, reward, done, {}

    def _finish_step(self, step, dt, prev=None):
        """
        Apply a physical step.

        If prev is None, the step is applied to the current grid, otherwise
        it is applied to prev, then saved to the grid.

        Returns state, reward, done.
        """

        if self.eps > 0.0:
            R = self.eps * self.lap()
            step += dt * R[g.ilo:g.ihi+1]
        if self.source is not None:
            self.source.update(dt, self.t + dt)
            step += dt * self.source.get_real()

        if prev is None:
            u_start = self.grid.get_real()
        else:
            u_start = prev
        self.grid.set(u_start + step)

        self.t += dt

        state = self.prep_state()

        self.steps += 1
        if self.steps >= self.episode_length:
            done = True
        else:
            done = False

        self.solution.update(dt, self.t)
        reward, force_done = self.calculate_reward()
        done = done or force_done

        if self.weno_solution is not None:
            self.weno_solution.update(dt, self.t)

        if np.isnan(state).any():
            raise Exception("NaN detected in state.")
        if np.isnan(reward).any():
            raise Exception("NaN detected in reward.")

        return state, reward, done


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
        self.grid.set(u_copy)

        self.t += dt

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
        self.grid.set(u_copy)

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

