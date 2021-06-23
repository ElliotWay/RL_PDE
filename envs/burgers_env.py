import os
import sys
from datetime import datetime

import gym
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import tensorflow as tf
from gym import spaces
from stable_baselines import logger

from envs.grid import Grid1d
from envs.source import RandomSource
from envs.solutions import PreciseWENOSolution, AnalyticalSolution
from envs.solutions import MemoizedSolution, OneStepSolution
import envs.weno_coefficients as weno_coefficients
from util.softmax_box import SoftmaxBox
from util.misc import create_stencil_indexes

#TODO (long-term):
# Rewrite this file. It's close to 2000 lines long, as of writing this comment, which is too many
# for one file. Probably the Abstract class should be separate, and only have e.g. the plot
# functions.

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
            init_params=None,
            fixed_step=0.0004, C=None, #C=0.5,
            weno_order=3, state_order=None, eps=0.0, srca=0.0, episode_length=250,
            analytical=False, precise_weno_order=None, precise_scale=1,
            reward_adjustment=1000, reward_mode=None,
            memoize=False,
            test=False):
        """
        Construct the Burgers environment.

        Parameters
        ----------
        xmin : float
            lower bound of the physical space
        xmax : float
            upper bound of the physical space
        nx : int
            number of discretized cells
        boundary : string
            type of boundary condition (periodic/outflow)
        init_type : string
            type of initial condition (various, see grid.py)
        fixed_step : float
            length of fixed timestep
        C : float
            CFL number for varying length timesteps (NOT CURRENTLY USED)
        weno_order : int
            Order of WENO approximation. Affects action and state space sizes.
        state_order : int
            Use to specify a state space that is wider than it otherwise should be.
        eps : float
            viscosity parameter
        srca : float
            source amplitude
        episode_length : int
            number of timesteps before the episode terminates
        analytical : bool
            whether to use an analytical solution instead of a numerical one
        precise_weno_order : int
            use to compute the weno solution with a higher weno order
        precise_scale : int
            use to compute the weno solution with more precise grid
        reward_adjustment : float
            adjust the reward squashing function - higher makes small changes to large rewards
            less important
        reward_mode : str
            composite string specifying the type of reward function
        memoize : bool
            use a memoized weno solution
        test : bool
            whether this is a strictly test environment (passed to Grid1d to ensure initial
            conditions are not randomized)
        """

        self.test = test
        
        self.reward_mode = self.fill_default_reward_mode(reward_mode)
        if reward_mode != self.reward_mode:
            print("Reward mode updated to '{}'.".format(self.reward_mode))

        self.nx = nx
        self.weno_order = weno_order
        if state_order is None:
            self.state_order = weno_order
        else:
            self.state_order = state_order
        self.ng = self.state_order + 1
        self.grid = Grid1d(xmin=xmin, xmax=xmax, nx=nx, ng=self.ng,
                           boundary=boundary, init_type=init_type,
                           deterministic_init=self.test)
        self.init_params = init_params

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
                    boundary=boundary, init_type=init_type, flux_function=self.burgers_flux, source=self.source,
                    eps=eps)

        if "one-step" in self.reward_mode:
            self.solution = OneStepSolution(self.solution, self.grid)
            if memoize:
                print("Note: can't memoize solution when using one-step reward.")
        elif memoize:
            self.solution = MemoizedSolution(self.solution)

        show_separate_weno = (self.analytical
                              or self.precise_weno_order != self.weno_order
                              or self.precise_nx != self.nx
                              or isinstance(self.solution, OneStepSolution))
        if show_separate_weno:
            self.weno_solution = PreciseWENOSolution(xmin=xmin, xmax=xmax, nx=nx, ng=self.ng,
                                                     precise_scale=1, precise_order=weno_order,
                                                     boundary=boundary, init_type=init_type,
                                                     flux_function=self.burgers_flux,
                                                     source=self.source, eps=eps)
            if memoize:
                self.weno_solution = MemoizedSolution(self.weno_solution)
        else:
            self.weno_solution = None

        self.previous_error = np.zeros_like(self.grid.get_full())

        self.fixed_step = fixed_step
        self.C = C  # CFL number
        self.eps = eps
        self.episode_length = episode_length
        self.reward_adjustment = reward_adjustment

        self._step_precision = int(np.ceil(np.log(1+self.episode_length) / np.log(10)))
        self._cell_index_precision = int(np.ceil(np.log(1+self.nx) / np.log(10)))

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
        """
        Reset the environment.

        This is the abstract class version. In a subclass, the overriding function should call this
        version, then return the initial state based on how the subclass is configured.

        Returns
        -------
        Nothing! However, the subclass versions should return the initial state.

        """
        self.grid.reset(params=self.init_params)
        if self.source is not None:
            self.source.reset()
        self.solution.reset(self.grid.init_params)
        if self.weno_solution is not None:
            self.weno_solution.reset(self.grid.init_params)

        self.state_history = [self.grid.get_full().copy()]
        self.action_history = []

        self.t = 0.0
        self.steps = 0
        self.previous_error = np.zeros_like(self.grid.get_full())

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

    @staticmethod
    def fill_default_time_vs_space(xmin, xmax, nx, dt, C, ep_length, time_max):
        approximate_max = 2.0
        if nx is None:
            if dt is None:
                dt = 0.0004
            dx = dt * approximate_max / C
            nx = int((xmax - xmin) / dx)
            #nx = max(nx, 2*order - 1)
        elif dt is None:
            dx = (xmax - xmin) / nx
            dt = C * dx / approximate_max

        if ep_length is None:
            if time_max is None:
                ep_length = 500
            else:
                ep_length = int(np.ceil(time_max / dt))

        return nx, dt, ep_length

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

    def force_state(self, state_grid):
        """
        Override the current state with something else.

        You should have a unusual reason for doing this, like if you need to copy the state from a
        different environment.
        state and action history will not make sense after calling this.

        Parameters
        ----------
        state_grid : ndarray
            Array of new state values. Should not include ghost values.
        """
        self.grid.set(state_grid)
        # Rewrite recent history.
        self.state_history[-1] = self.grid.get_full().copy()

    def get_state(self, timestep=None, location=None, full=True):
        assert timestep is None or location is None

        if timestep is None and location is None:
            state = self.grid.get_full() if full else self.grid.get_real()
        elif timestep is not None:
            state = np.array(self.state_history)[timestep, :]
            if not full:
                state = state[self.ng:-self.ng]
        else:
            state = np.array(self.state_history)[:, location]
        return state

    def get_solution_state(self, timestep=None, location=None, full=True):
        assert timestep is None or location is None

        #TODO: Does it make sense to return the self.weno_solution state instead if using a
        # one-step reward? (That is, a OneStepSolution?)

        if timestep is None and location is None:
            state = self.solution.get_full() if full else self.solution.get_real()
        elif timestep is not None:
            state = np.array(self.solution.get_state_history())[timestep, :]
            if not full:
                state = state[self.ng:-self.ng]
        else:
            state = np.array(self.solution.get_state_history())[:, location]
        return state

    def get_error(self, timestep=None, location=None, full=True):
        return (self.get_state(timestep, location, full) 
                - self.get_solution_state(timestep, location, full))

    def compute_l2_error(self, timestep=None):
        """
        Compute the L2 error between the solution and the state at a given timestep.
        By default, the current timestep is used.

        Parameters
        ----------
        timestep : int (or string)
            Timestep for which to calculate the L2 error. Passing "all" for timestep will instead
            calculate the L2 error at every timestep, and return them as a list.


        Returns
        -------
        l2_error : float (or list of floats)
            The L2 error, or list of errors if "all" is passed to timestep.
        """
        if timestep == "all":
            l2_errors = []
            for step in range(len(self.state_history)):
                l2_errors.append(self.compute_l2_error(step))
            return l2_errors

        else:
            error = self.get_error(timestep=timestep, full=False)
            l2_error = np.sqrt(self.grid.dx * np.sum(np.square(error)))
            return l2_error

    def get_action(self, timestep=None, location=None):
        assert timestep is None or location is None

        action_history = np.array(self.action_history)
        if timestep is None and location is None:
            action = action_history[-1]
        elif timestep is not None:
            action = action_history[timestep]
        else:
            action = action_history[:, location]
        return action

    def get_solution_action(self, timestep=None, location=None):
        assert self.solution.is_recording_actions()
        assert timestep is None or location is None

        solution_action_history = np.array(self.solution.get_action_history())
        if timestep is None and location is None:
            action = solution_action_history[-1]
        elif timestep is not None:
            action = solution_action_history[timestep]
        else:
            action = solution_action_history[:, location]
        return action

    def plot_state(self,
            timestep=None, location=None,
            plot_error=False,
            suffix=None, title=None,
            fixed_axes=False, no_x_borders=False, show_ghost=True,
            state_history=None, solution_state_history=None, weno_state_history=None,
            history_includes_ghost=True):
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
        state_history [, solution_state_history, weno_state_history] : ndarray
            Override the current state histories with a different set. Useful if, for example, you
            copied the history from an earlier episode.
            If using state_history, you must also use solution_state_history. weno_state_history is
            optional and only necessary if solution_state_history is something different.
        history_includes_ghost : bool
            Whether the overriding histories include the ghost cells. history_includes_ghost=False
            overrides show_ghost to False.
        """

        assert (timestep is None or location is None), "Can't plot state at both a timestep and a location."

        override_history = (state_history is not None)

        fig = plt.figure()

        error_or_state = "error" if plot_error else "state"

        if location is None and timestep is None:
            if not override_history:
                state_history = self.grid.get_full().copy()
                solution_state_history = self.solution.get_full().copy()
                if self.weno_solution is not None:
                    weno_state_history = self.weno_solution.get_full().copy()
                else:
                    weno_state_history = None

                num_steps = self.steps
                actual_time = self.t
            else:
                num_steps = len(state_history)
                actual_time = num_steps * self.fixed_step

            if title is None:
                title = ("{} at t = {:.4f} (step {:0" + str(self._step_precision) + "d})").format(
                        error_or_state, actual_time, self.steps)
            if suffix is None:
                suffix = ("_step{:0" + str(self._step_precision) + "}").format(num_steps)
        else:
            if not override_history:
                state_history = np.array(self.state_history)
                solution_state_history = self.solution.get_state_history().copy() \
                        if self.solution.is_recording_state() else None
                weno_state_history = self.weno_solution.get_state_history().copy() \
                        if self.weno_solution is not None and self.weno_solution.is_recording_state() else None

            if location is not None:
                if not override_history or history_includes_ghost:
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
                        title = ("{} at t = {:.4f} (step {:0" + str(self._step_precision) + "d})").format(
                                error_or_state, actual_time, timestep)
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
            if show_ghost and (not override_history or history_includes_ghost):
                # The ghost arrays slice off one real point so the line connects to the real points.
                num_ghost_points = self.ng + 1

                ghost_x_left = self.grid.x[:num_ghost_points]
                ghost_x_right = self.grid.x[-num_ghost_points:]

                plt.plot(ghost_x_left, state_history[:num_ghost_points], ls='-', color=self.agent_ghost_color)
                plt.plot(ghost_x_right, state_history[-num_ghost_points:], ls='-', color=self.agent_ghost_color)

                if solution_state_history is not None:
                    plt.plot(ghost_x_left, solution_state_history[:num_ghost_points],
                            ls='-', color=self.true_ghost_color)
                    plt.plot(ghost_x_right, solution_state_history[-num_ghost_points:],
                            ls='-', color=self.true_ghost_color)

                if weno_state_history is not None:
                    plt.plot(ghost_x_left, weno_state_history[:num_ghost_points],
                            ls='-', color=self.weno_ghost_color)
                    plt.plot(ghost_x_right, weno_state_history[-num_ghost_points:],
                            ls='-', color=self.weno_ghost_color)

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
            if weno_state_history is not None:
                if self.precise_weno_order != self.weno_order or self.precise_nx != self.nx:
                    true_label = "WENO (order={}, res={})".format(self.precise_weno_order, self.precise_nx)
                    weno_label = "WENO (order={}, res={})".format(self.weno_order, self.nx)
                elif isinstance(self.solution, OneStepSolution):
                    true_label = "WENO (one-step)"
                    weno_label = "WENO (full)"
                else:
                    print("Why do we have both a solution and a WENO solution?"
                            + " I don't know what labels to use.")
                    true_label = "WENO"
                    weno_label = "WENO"
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

    def plot_state_evolution(self,
            num_states=10, full_true=False, no_true=False, plot_error=False, plot_weno=False,
            suffix="", title=None,
            state_history=None, solution_state_history=None, weno_state_history=None,
            ):
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
            the same number of timesteps as the RL solution. (This may
            be confusing to interpret.)
        no_true : bool
            Set False by default, which plots the true solution. Set True to
            ONLY plot the RL solution, if you don't care about the true solution.
            Also useful to plot evolution of the true solution itself.
        plot_weno : bool
            Plot the separate WENO solution, if it exists and it is appropriate to do so.
            Set False by default.
        suffix : string
            The plot will be saved to burgers_evolution_state{suffix}.png
            (or burgers_evolution_error{suffix}.png). There is no suffix by
            default.
        title : string
            Title for the plot. There is no title by default.
        state_history [, solution_state_history, weno_state_history] : ndarray
            Override the current state histories with a different set. Useful if, for example, you
            copied the history from an earlier episode.
        """

        override_history = (state_history is not None)

        fig = plt.figure()

        x_values = self.grid.x[self.ng:-self.ng]

        if not override_history:
            state_history = np.array(self.state_history)[:, self.ng:-self.ng]
            solution_state_history = None
            weno_state_history = None

        # Indexes into the state history. There are num_states+1 indices, where the first
        # is always 0 the last is always len(state_history)-1, and the rest are evenly
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
        true = None
        weno = None
        if not plot_error and not no_true:
            if not override_history:
                if not isinstance(self.solution, OneStepSolution) and self.solution.is_recording_state():
                #if isinstance(self.solution, OneStepSolution):
                    solution_state_history = np.array(self.solution.get_state_history())[:, self.ng:-self.ng]
                    if (plot_weno and self.weno_solution is not None and
                                    self.weno_solution.is_recording_state()):
                        weno_state_history = np.array(
                                self.weno_solution.get_state_history())[:, self.ng:-self.ng]

                elif self.weno_solution is not None and self.weno_solution.is_recording_state():
                    solution_state_history = np.array(
                                self.weno_solution.get_state_history())[:, self.ng:-self.ng]

            if solution_state_history is not None:
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

                if weno_state_history is not None:
                    assert len(state_history) == len(weno_state_history), "History mismatch."
                    if full_true:
                        weno_rgb = matplotlib.colors.to_rgb(self.weno_color)
                        weno_color_sequence = color_sequence(start_rgb, weno_rgb, num_states)
                        sliced_solution_history = weno_state_history[slice_indexes]
                        
                        for state_values, color in zip(sliced_solution_history[1:-1], weno_color_sequence[1:-1]):
                            plt.plot(x_values, state_values, ls='-', linewidth=1, color=color)
                        weno = plt.plot(x_values, weno_state_history[-1],
                                        ls='-', linewidth=1, color=self.weno_color)
                    else:
                        weno = plt.plot(x_values, weno_state_history[-1], 
                                        ls='-', linewidth=4, color=self.weno_color)

        if plot_error:
            if not override_history:
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
            if weno is not None:
                plots.append(weno[0])
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

            actual_location = self.grid.x[location] - self.grid.dx/2
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

    @staticmethod
    def fill_default_reward_mode(reward_mode_arg):
        reward_mode = "" if reward_mode_arg is None else reward_mode_arg

        if (not "full" in reward_mode
                and not "change" in reward_mode
                and not "one-step" in reward_mode):
            if "l2d" in reward_mode:
                reward_mode += "_full"
            else:
                reward_mode += "_full"

        if (not "adjacent" in reward_mode
                and not "stencil" in reward_mode):
            if "l2d" in reward_mode:
                reward_mode += "_stencil"
            else:
                reward_mode += "_adjacent"

        if (not "avg" in reward_mode
                and not "max" in reward_mode
                and not "L2" in reward_mode):
            if "l2d" in reward_mode:
                reward_mode += "_max"
            else:
                reward_mode += "_avg"

        if (not "squash" in reward_mode):
            if "l2d" in reward_mode:
                reward_mode += "_logsquash"
            else:
                reward_mode += "_arctansquash"

        return reward_mode
    
    def calculate_reward(self):
        """ Reward calculation based on the error between grid and solution. """

        done = False

        # Use difference with WENO actions instead. (Might be useful for testing.)
        if "wenodiff" in self.reward_mode:
            last_action = self.action_history[-1].copy()

            if self.solution.is_recording_actions():
                weno_action = self.solution.get_action_history()[-1].copy()
            elif self.weno_solution is not None:
                assert self.weno_solution.is_recording_actions()
                weno_action = self.weno_solution.get_action_history()[-1].copy()
            else:
                raise Exception("AbstractBurgersEnv: reward_mode problem")
            action_diff = weno_action - last_action
            action_diff = action_diff.reshape((len(action_diff), -1))
            if "L1" in self.reward_mode:
                error = np.sum(np.abs(action_diff), axis=-1)
            elif "L2" in self.reward_mode:
                error = np.sqrt(np.sum(action_diff**2, axis=-1))
            else:
                raise Exception("AbstractBurgersEnv: reward_mode problem")

            return -error, done


        # Use error with the "true" solution as the reward.
        error = self.solution.get_full() - self.grid.get_full()

        if "full" in self.reward_mode or "one-step" in self.reward_mode:
            # one-step is handled by the solution
            pass
        # Use the difference in error as a reward instead of the full reward with the solution.
        elif "change" in self.reward_mode:
            previous_error = self.previous_error
            self.previous_error = error
            error = (error - previous_error)
        else:
            raise Exception("AbstractBurgersEnv: reward_mode problem")

        error = np.abs(error)

        # Clip tiny errors and enhance extreme errors.
        if "clip" in self.reward_mode:
            error[error < 0.001] = 0
            error[error > 0.1] *= 10

        # Average of error in two adjacent cells.
        if "adjacent" in self.reward_mode and "avg" in self.reward_mode:
            combined_error = (error[self.ng-1:-self.ng] + error[self.ng:-(self.ng-1)]) / 2
        # Combine error across the WENO stencil.
        # (NOT the state stencil i.e. self.state_order * 2 - 1, even if we are using a wide state.)
        elif "stencil" in self.reward_mode:
            stencil_indexes = create_stencil_indexes(
                    stencil_size=(self.weno_order * 2 - 1),
                    num_stencils=(self.nx + 1),
                    offset=(self.ng - self.weno_order))
            error_stencils = error[stencil_indexes]
            if "max" in self.reward_mode:
                combined_error = np.amax(error_stencils, axis=-1)
            elif "avg" in self.reward_mode:
                combined_error = np.mean(error_stencils, axis=-1)
            elif "L2" in self.reward_mode:
                combined_error = np.sqrt(np.sum(error_stencils**2, axis=-1))
            else:
                raise Exception("AbstractBurgersEnv: reward_mode problem")
        else:
            raise Exception("AbstractBurgersEnv: reward_mode problem")

        # Squash reward.
        if "nosquash" in self.reward_mode:
            max_penalty = 1e7
            reward = -combined_error
        elif "logsquash" in self.reward_mode:
            max_penalty = 1e7
            reward = -np.log(combined_error + 1e-30)
        elif "arctansquash" in self.reward_mode:
            max_penalty = np.pi / 2
            if "noadjust" in self.reward_mode:
                reward = np.arctan(-combined_error)
            else:
                # The constant controls the relative importance of small rewards compared to large rewards.
                # Towards infinity, all rewards (or penalties) are equally important.
                # Towards 0, small rewards are increasingly less important.
                # An alternative to arctan(C*x) with this property would be x^(1/C).
                reward = np.arctan(self.reward_adjustment * -combined_error)
        else:
            raise Exception("AbstractBurgersEnv: reward_mode problem")

        # Conservation-based reward.
        # Doesn't work (always 0), but a good idea. We'll try this again eventually.
        # reward = -np.log(np.sum(rhs[g.ilo:g.ihi+1]))

        # Give a penalty and end the episode if we're way off.
        #if np.max(state) > 1e7 or np.isnan(np.max(state)): state possibly made more sense here?
        if np.max(error) > 1e7 or np.isnan(np.max(error)):
            reward -= max_penalty * (self.episode_length - self.steps)
            done = True

        #print("reward:", reward)

        return reward, done

    @tf.function
    def tf_prep_state(self, state):
        """
        Function that converts the real physical state to the RL state perceived by the agent, but
        written as a Tensorflow function so that it can be inserted into a network and
        backpropagated through.

        Parameters
        ----------
        state : Tensor
            Representation of the real physical state. This should have no ghost cells.

        Returns
        -------
        rl_state : Tensor
            Version of the state perceived by the agent. This SHOULD have ghost cells.
        """
        raise NotImplementedError()

    @tf.function
    def tf_integrate(self, args):
        """
        Function that calculates the next real physical state based on the previous state and
        agent's action, but written as a Tensorflow function so that it can be inserted into a
        network and backpropagated through.

        Parameters
        ----------
        args : tuple
            The tuple should be made up of (real_state, rl_state, rl_action).
            (Needs to be a tuple so this function works with tf.map_fn.)
            real_state : Tensor
                Representation of the real physical state. This should have no ghost cells.
            rl_state : Tensor
                Version of the state perceived by the agent (as returned by tf_prep_state()). This
                SHOULD have ghost cells.
            rl_action : Tensor
                Action from the agent that corresponds to rl_state.

        Returns
        -------
        next_real_state : Tensor
            The real physical state on the following timestep. This should also have no ghost
            cells.
        """

        raise NotImplementedError()

    @tf.function
    def tf_calculate_reward(self, args):
        """
        Function that calculates the reward based on the difference between the physical state
        reached by following actions suggested by the agent and the physical state that would be
        reached by following WENO, but written as a Tensorflow function so that it can be inserted
        into a network and backpropagated through.

        Parameters
        ----------
        args : tuple
            The tuple should be made up of (real_state, rl_state, rl_action, next_real_state).
            (Needs to be a tuple so this function works with tf.map_fn.)
            real_state : Tensor
                Representation of the real physical state. This should have no ghost cells.
            rl_state : Tensor
                Version of the state perceived by the agent (as returned by tf_prep_state()). This
                SHOULD have ghost cells.
            rl_action : Tensor
                Action from the agent that corresponds to rl_state.
            next_real_state : Tensor
                The real physical state on the following timestep (as returned by tf_integrate()).
                This should also have no ghost cells.

        Returns
        -------
        reward : Tensor
            Reward for the agent at each location.
        """
        raise NotImplementedError()
 
    def close(self):
        # Delete references for easier garbage collection.
        self.grid = None
        self.solution = None
        self.weno_solution = None
        self.state_history = []
        self.action_history = []

    def evolve(self):
        """
        Evolve the environment using the solution, instead of passing actions.

        The state will be an identical copy of the solution's state.
        Does not work with the 'one-step' error - this makes the solution dependent on the state,
        but evolving this way requires the state to depend on the solution.
        """
        assert (not isinstance(self.solution, OneStepSolution)), \
            "Can't evolve with one-step solution."

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

    def __init__(self, follow_solution=False, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.follow_solution = follow_solution

        self.action_space = SoftmaxBox(low=0.0, high=1.0, 
                                       shape=(self.grid.real_length() + 1, 2, self.weno_order),
                                       dtype=np.float32)
        self.observation_space = spaces.Box(low=-1e7, high=1e7,
                                            shape=(self.grid.real_length() + 1, 2, 2 * self.state_order - 1),
                                            dtype=np.float64)
       
        self.solution.set_record_state(True)
        if self.weno_solution is not None:
            self.weno_solution.set_record_state(True)

        if self.weno_solution is not None:
            self.weno_solution.set_record_actions("weno")
        elif not self.analytical:
            self.solution.set_record_actions("weno")

        self.dt = self.timestep()
        self.k1 = self.k2 = self.k3 = self.u_start = None
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

        fp_stencil_indexes = create_stencil_indexes(stencil_size=self.state_order * 2 - 1,
                                                    num_stencils=g.real_length() + 1,
                                                    offset=g.ng - self.state_order)
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
        super().reset()

        self.rk_state = 1

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

        action and state are only recorded on the 4th call.

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

            self.action_history.append(action)

            k4 = self.dt * self.rk_substep_weno(action)
            step = (self.k1 + 2*(self.k2 + self.k3) + k4) / 6

            if isinstance(self.solution, PreciseWENOSolution):
                self.solution.set_rk4(True)
            if self.weno_solution is not None:
                self.weno_solution.set_rk4(True)

            state, reward, done = self._finish_step(step, self.dt, prev=self.u_start)

            if isinstance(self.solution, PreciseWENOSolution):
                self.solution.set_rk4(False)
            if self.weno_solution is not None:
                self.weno_solution.set_rk4(False)

            self.state_history.append(self.grid.get_full().copy())

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

        self.action_history.append(action)

        # Find a way to pass this to agent if dt varies.
        dt = self.timestep()

        step = dt * self.rk_substep_weno(action)

        state, reward, done = self._finish_step(step, dt)

        self.state_history.append(self.grid.get_full().copy())

        return state, reward, done, {}

    def _finish_step(self, step, dt, prev=None):
        """
        Apply a physical step.

        If prev is None, the step is applied to the current grid, otherwise
        it is applied to prev, then saved to the grid.

        Returns state, reward, done.
        """
        # Update solution first, in case it depends on our current state.
        self.solution.update(dt, self.t)

        if self.weno_solution is not None:
            self.weno_solution.update(dt, self.t)

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

        self.steps += 1
        if self.steps >= self.episode_length:
            done = True
        else:
            done = False

        reward, force_done = self.calculate_reward()
        done = done or force_done

        # Set the state to the solution after calculating
        # the reward. Useful for testing?
        if self.follow_solution:
            self.grid.set(self.solution.get_real())

        state = self.prep_state()

        if np.isnan(state).any():
            raise Exception("NaN detected in state.")
        if np.isnan(reward).any():
            raise Exception("NaN detected in reward.")

        return state, reward, done

    @tf.function
    def tf_prep_state(self, state):
        # Note the similarity to prep_state() above.

        # state (the real physical state) does not have ghost cells, but agents operate on a stencil
        # that can spill beyond the boundary, so we need to add ghost cells to create the rl_state.

        # TODO: get boundary from initial condition, somehow, as diff inits have diff bounds
        #  Would need to use tf.cond so graph can handle different boundaries at runtime.
        ghost_size = tf.constant(self.ng, shape=(1,))
        if self.grid.boundary == "outflow":
            # This implementation assumes that state is a 1-D Tensor of scalars.
            # In the future, if we expand to multiple dimensions, then it won't be, so this will need
            # to be changed (probably use tf.tile instead).
            # Not 100% sure tf.fill can be used this way.
            left_ghost = tf.fill(ghost_size, state[0])
            right_ghost = tf.fill(ghost_size, state[-1])
            full_state = tf.concat([left_ghost, state, right_ghost], axis=0)
        elif self.grid.boundary == "periodic":
            left_ghost = state[-ghost_size[0]:]
            right_ghost = state[:ghost_size[0]]
            full_state = tf.concat([left_ghost, state, right_ghost], axis=0)
        else:
            raise NotImplementedError()

        # Compute flux.
        flux = 0.5 * (full_state ** 2)

        alpha = tf.reduce_max(tf.abs(flux))

        flux_plus = (flux + alpha * full_state) / 2
        flux_minus = (flux - alpha * full_state) / 2

        #TODO Could change things to use traditional convolutions instead.
        # Maybe if this whole thing actually works.

        plus_indexes = create_stencil_indexes(
                        stencil_size=(self.weno_order * 2 - 1),
                        num_stencils=(self.nx + 1),
                        offset=(self.ng - self.weno_order))
        minus_indexes = plus_indexes + 1
        minus_indexes = np.flip(minus_indexes, axis=-1)

        plus_stencils = tf.gather(flux_plus, plus_indexes)
        minus_stencils = tf.gather(flux_minus, minus_indexes)

        # Stack together into rl_state.
        # Stack on dim 1 to keep location dim 0.
        rl_state = tf.stack([plus_stencils, minus_stencils], axis=1)

        return rl_state

    @tf.function
    def tf_integrate(self, args):
        real_state, rl_state, rl_action = args

        # Note that real_state does not contain ghost cells here, but rl_state DOES (and rl_action has
        # weights corresponding to the ghost cells).

        plus_stencils = rl_state[:, 0]
        minus_stencils = rl_state[:, 1]

        #Note the similarity in this block to weno_i_stencils_batch().
        a_mat = weno_coefficients.a_all[self.weno_order]
        a_mat = np.flip(a_mat, axis=-1)
        a_mat = tf.constant(a_mat, dtype=rl_state.dtype)
        sub_stencil_indexes = create_stencil_indexes(stencil_size=self.weno_order,
                num_stencils=self.weno_order)
        # Here axis 0 is location/batch and axis 1 is stencil,
        # so to index substencils we gather on axis 1.
        plus_sub_stencils = tf.gather(plus_stencils, sub_stencil_indexes, axis=1)
        minus_sub_stencils = tf.gather(minus_stencils, sub_stencil_indexes, axis=1)
        plus_interpolated = tf.reduce_sum(a_mat * plus_sub_stencils, axis=-1)
        minus_interpolated = tf.reduce_sum(a_mat * minus_sub_stencils, axis=-1)

        plus_action = rl_action[:, 0]
        minus_action = rl_action[:, 1]

        fpr = tf.reduce_sum(plus_action * plus_interpolated, axis=-1)
        fml = tf.reduce_sum(minus_action * minus_interpolated, axis=-1)

        reconstructed_flux = fpr + fml

        derivative_u_t = (reconstructed_flux[:-1] - reconstructed_flux[1:]) / self.grid.dx

        #TODO implement RK4?

        step = self.dt * derivative_u_t

        #TODO implement viscosity and random source?
        if self.eps != 0.0:
            raise NotImplementedError("Viscosity has not been implemented in global backprop.")
        if self.source != None:
            raise NotImplementedError("External source has not been implemented"
                    + " in global backprop.")
        
        new_state = real_state + step
        return new_state

    # TODO: modify this so less is in subclass? The reward section could go in the abstract class,
    # but this needs to be here because of how we calculate the WENO step. Not a high priority.
    # (And anyway, if we're refactoring, this should probably be the one canonical function instead
    # of having TF and non-TF versions.)
    @tf.function
    def tf_calculate_reward(self, args):
        real_state, rl_state, rl_action, next_real_state = args
        # Note that real_state and next_real_state do not include ghost cells, but rl_state does.

        # This section is adapted from agents.py#StandardWENOAgent._weno_weights_batch()
        C_values = weno_coefficients.C_all[self.weno_order]
        C_values = tf.constant(C_values, dtype=real_state.dtype)
        sigma_mat = weno_coefficients.sigma_all[self.weno_order]
        sigma_mat = tf.constant(sigma_mat, dtype=real_state.dtype)

        fp_stencils = rl_state[:, 0]
        fm_stencils = rl_state[:, 1]

        sub_stencil_indexes = create_stencil_indexes(stencil_size=self.weno_order,
                num_stencils=self.weno_order)

        # Here axis 0 is location/batch and axis 1 is stencil,
        # so to index substencils we gather on axis 1.
        sub_stencils_fp = tf.reverse(tf.gather(fp_stencils, sub_stencil_indexes, axis=1),
                axis=(-1,)) # For some reason, axis MUST be iterable, can't be scalar -1.
        sub_stencils_fm = tf.reverse(tf.gather(fm_stencils, sub_stencil_indexes, axis=1),
                axis=(-1,))

        q_squared_fp = sub_stencils_fp[:, :, None, :] * sub_stencils_fp[:, :, :, None]
        q_squared_fm = sub_stencils_fm[:, :, None, :] * sub_stencils_fm[:, :, :, None]

        beta_fp = tf.reduce_sum(sigma_mat * q_squared_fp, axis=(-2, -1))
        beta_fm = tf.reduce_sum(sigma_mat * q_squared_fm, axis=(-2, -1))

        epsilon = tf.constant(1e-16)
        alpha_fp = C_values / (epsilon + beta_fp ** 2)
        alpha_fm = C_values / (epsilon + beta_fm ** 2)

        weights_fp = alpha_fp / (tf.reduce_sum(alpha_fp, axis=-1)[:, None])
        weights_fm = alpha_fm / (tf.reduce_sum(alpha_fm, axis=-1)[:, None])
        weno_action = tf.stack([weights_fp, weights_fm], axis=1)
        # end adaptation of _weno_weights_batch()

        weno_next_real_state = self.tf_integrate((real_state, rl_state, weno_action))

        # This section is adapted from AbstactBurgersEnv.calculate_reward()
        if "wenodiff" in self.reward_mode:
            last_action = rl_action

            action_diff = weno_action - last_action
            partially_flattened_shape = (self.action_space.shape[0],
                    np.prod(self.action_space.shape[1:]))
            action_diff = tf.reshape(action_diff, partially_flattened_shape)
            if "L1" in self.reward_mode:
                error = tf.reduce_sum(tf.abs(action_diff), axis=-1)
            elif "L2" in self.reward_mode:
                error = tf.sqrt(tf.reduce_sum(action_diff**2, axis=-1))
            else:
                raise Exception("reward_mode problem")

            return -error

        error = weno_next_real_state - next_real_state

        if "one-step" in self.reward_mode:
            pass
            # This version is ALWAYS one-step - the others are tricky to implement in TF.
        elif "full" in self.reward_mode:
            raise Exception("Reward mode 'full' invalid - only 'one-step' reward implemented"
                " in Tensorflow functions.")
        elif "change" in self.reward_mode:
            raise Exception("Reward mode 'change' invalid - only 'one-step' reward implemented"
                " in Tensorflow functions.")
        else:
            raise Exception("reward_mode problem")

        if "clip" in self.reward_mode:
            raise Exception("Reward clipping not implemented in Tensorflow functions.")

        # TODO same thing as in tf_prep_state - need to handle changes to boundary at runtime

        # Average of error in two adjacent cells.
        if "adjacent" in self.reward_mode and "avg" in self.reward_mode:
            error = tf.abs(error)
            combined_error = (error[:-1] + error[1:]) / 2
            if self.grid.boundary == "outflow":
                # Error beyond boundaries will be identical to error at edge, so average error
                # at edge interfaces is just the error at the edge.
                combined_error = tf.concat([[error[0]], combined_error, [error[-1]]], axis=0)
            elif self.grid.boundary == "periodic":
                # With a periodic environment, the first and last interfaces are actually the
                # same interface.
                edge_error = (error[0] + error[-1]) / 2
                combined_error = tf.concat([[edge_error], combined_error, [edge_error]], axis=0)
            else:
                raise NotImplementedError()
        # Combine error across the stencil.
        elif "stencil" in self.reward_mode:
            # This is trickier and we need to expand the error into the ghost cells.
            ghost_size = tf.constant(self.ng, shape=(1,))
            if self.grid.boundary == "outflow":
                left_error = tf.fill(ghost_size, error[0])
                right_error = tf.fill(ghost_size, error[-1])
            elif self.grid.boundary == "periodic":
                left_error = error[-ghost_size[0]:]
                right_error = error[:ghost_size[0]]
            else:
                raise NotImplementedError()
            full_error = tf.concat([left_error, error, right_error])
            
            stencil_indexes = create_stencil_indexes(
                    stencil_size=(self.weno_order * 2 - 1),
                    num_stencils=(self.nx + 1),
                    offset=(self.ng - self.weno_order))
            error_stencils = full_error[stencil_indexes]
            if "max" in self.reward_mode:
                error_stencils = tf.abs(error_stencils)
                combined_error = tf.reduce_max(error_stencils, axis=-1)
            elif "avg" in self.reward_mode:
                error_stencils = tf.abs(error_stencils)
                combined_error = tf.reduce_mean(error_stencils, axis=-1)
            elif "L2" in self.reward_mode:
                combined_error = tf.sqrt(tf.reduce_sum(error_stencils**2, axis=-1))
            elif "L1" in self.reward_mode:
                error_stencils = tf.abs(error_stencils)
                combined_error = tf.reduce_sum(error_stencils, axis=-1)
            else:
                raise Exception("reward_mode problem")
        else:
            raise Exception("reward_mode problem")

        # Squash reward.
        if "nosquash" in self.reward_mode:
            reward = -combined_error
        elif "logsquash" in self.reward_mode:
            epsilon = tf.constant(1e-16)
            reward = -tf.log(combined_error + epsilon)
        elif "arctansquash" in self.reward_mode:
            if "noadjust" in self.reward_mode:
                reward = tf.atan(-combined_error)
            else:
                reward_adjustment = tf.constant(self.reward_adjustment, dtype=real_state.dtype)
                # The constant controls the relative importance of small rewards compared to large rewards.
                # Towards infinity, all rewards (or penalties) are equally important.
                # Towards 0, small rewards are increasingly less important.
                # An alternative to arctan(C*x) with this property would be x^(1/C).
                reward = tf.atan(reward_adjustment * -combined_error)
        else:
            raise Exception("reward_mode problem")

        # end adaptation of calculate_reward().

        return reward

class DiscreteWENOBurgersEnv(WENOBurgersEnv):
    def __init__(self, actions_per_dim=2, *args, **kwargs):
        super().__init__(*args, **kwargs)

        action_list = []
        action_list.append(np.zeros(self.weno_order, dtype=np.int32))

        # Enumerate combinations of integers that add up to actions_per_dim.
        # e.g. [0,1,2], [1,1,1], [3,0,0] for 3 actions per dimension.
        for dim in range(self.weno_order-1):
            new_actions = []
            for old_action in action_list:
                sum_so_far = np.sum(old_action)
                for i in range(actions_per_dim - sum_so_far):
                    new_action = old_action.copy()
                    new_action[dim] = i
                    new_actions.append(new_action)
            action_list = new_actions

        # Set the last action so that the action adds up to actions_per_dim.
        for action in action_list:
            remaining = actions_per_dim - 1 - np.sum(action)
            action[-1] = remaining

        # Normalize so the actions add up to 1 instead.
        self.action_list = np.array(action_list) / (actions_per_dim - 1)

        # At each interface, there are N^2 possible actions, because the + and -
        # directions have the same N options available, and we should accept any
        # combination.
        self.action_space = spaces.MultiDiscrete(
                    [len(action_list)**2] * ((self.grid.real_length() + 1)))

        # The observation space is declared in WENOBurgersEnv.

    def reset(self):
        return super().reset()

    def step(self, action):
        plus_actions = action // len(self.action_list)
        minus_actions = action % len(self.action_list)

        combined_actions = np.vstack((plus_actions, minus_actions)).transpose()

        selected_actions = self.action_list[combined_actions]

        return super().step(selected_actions)

class SplitFluxBurgersEnv(AbstractBurgersEnv):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.action_space = SoftmaxBox(low=-np.inf, high=np.inf,
                                       shape=(self.grid.real_length() + 1, 2, 2 * self.weno_order - 1),
                                       dtype=np.float32)
        self.observation_space = spaces.Box(low=-1e7, high=1e7,
                                            shape=(self.grid.real_length() + 1, 2, 2 * self.state_order - 1),
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

        fp_stencil_indexes = create_stencil_indexes(stencil_size=self.state_order * 2 - 1,
                                                    num_stencils=g.real_length() + 1,
                                                    offset=g.ng - self.state_order)
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
        super().reset()

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

        self.solution.update(dt, self.t)

        if self.weno_solution is not None:
            self.weno_solution.update(dt, self.t)

        u_copy += dt * rhs
        self.grid.set(u_copy)

        self.t += dt

        state = self.prep_state()

        self.steps += 1
        if self.steps >= self.episode_length:
            done = True

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
                                       dtype=np.float32)
        self.observation_space = spaces.Box(low=-1e7, high=1e7,
                                            shape=(self.grid.real_length() + 1, 2 * self.state_order),
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
        stencil_indexes = create_stencil_indexes(stencil_size=self.state_order * 2,
                                                 num_stencils=g.real_length() + 1,
                                                 offset=g.ng - self.state_order
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
        super().reset()

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

        self.solution.update(dt, self.t)
        if self.weno_solution is not None:
            self.weno_solution.update(dt, self.t)

        u_copy += dt * rhs
        self.grid.set(u_copy)

        # update the solution time
        self.t += dt

        # Calculate new RL state.
        state = self.prep_state()

        self.steps += 1
        if self.steps >= self.episode_length:
            done = True

        reward, force_done = self.calculate_reward()
        done = done or force_done


        if np.isnan(state).any():
            raise Exception("NaN detected in state.")
        if np.isnan(reward).any():
            raise Exception("NaN detected in reward.")

        return state, reward, done, {}

