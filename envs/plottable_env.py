import os

import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.ticker import LinearLocator
import matplotlib.animation as animation
import numpy as np

from util import sb_logger as logger
from util import plots

from envs.abstract_pde_env import AbstractPDEEnv
from envs.solutions import OneStepSolution

class Plottable1DEnv(AbstractPDEEnv):
    """
    Extension of PDE environment with plotting functions for 1D scalars.

    Can probably be extended to plotting vectors without too much reworking.
    Note that this is an abstract class - you can't declare a Plottable1DEnv.
    A subclass should extend this (and possibly other classes), then implement the requirements
    of subclasses of AbstractPDEEnv.
    """
    metadata = {'render.modes': ['file']}

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self._state_axes = None
        self._error_axes = None
        self._action_axes = None
        self._action_labels = None


        # We don't know whether self.weno_solution is declared or not until later, as it might be
        # declared by a child class, or a sibling class, so may not be declared by this point.
        self.weno_color = None
        self.weno_ghost_color = None
        self.true_color = None
        self.true_ghost_color = None
        self.agent_color = None
        self.agent_ghost_color = None
        # ghost green: "#94d194"

    def set_colors(self):
        if self.true_color is None:
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

    def render(self, mode='file', **kwargs):
        if mode is None:
            return
        if "file" in mode:
            return self.plot_state(**kwargs)
        else:
            raise Exception("Plottable1DEnv: \"{}\" render mode not recognized".format(mode))

    def euler_state_conversion(self, state):
        epsilon = 1e-16  # add a small number to avoid division by zero
        original_shape = list(state.shape)
        original_shape[0] += 1  # add a dimension for e
        original_state = np.zeros(shape=original_shape)
        original_state[0] = state[0]  # rho
        original_state[1] = state[1] / (state[0] + epsilon)  # v
        original_state[3] = (state[2] - original_state[0] * original_state[1] ** 2 / 2) / (original_state[0] + epsilon)
        original_state[2] = (self.eos_gamma - 1) * (state[2] - original_state[0] * original_state[1] ** 2 / 2)
        return original_state

    def save_state(self,
            timestep=None, location=None,
            use_error=False,
            suffix=None,
            show_ghost=False,
            state_history=None, solution_state_history=None,
            history_includes_ghost=True, silent=False):
        """
        Save the environment state to a csv file. Only the main RL state is saved, not the solution
        state.

        Parameters
        ----------
        timestep : int
            Timestep of the state to save. By default, use the most recent timestep.
        location : int
            Index of location of the state to save
        use_error : bool
            Save the error between the state and the solution state instead.
        suffix : string
            The plot will be saved to burgers_state_{suffix}.png (or burgers_error_{suffix}.png).
            By default, the timestep/location is used for the suffix.
        show_ghost : bool
            Save the ghost cells in addition to the 'real' cells.
        state_history [, solution_state_history] : ndarray
            Override the current state histories with different states.
        history_includes_ghost : bool
            Whether the overriding state history includes ghost cells. history_includes_ghost=False
            overrides show_ghost to False.
        silent : bool
            If False, print a message saying where the data was saved to.
        """
        assert (timestep is None or location is None), "Can't save state at both a timestep and a location."

        if 'Euler' in str(self):
            eqn_type = 'euler'
            ylabels = ['rho', 'u', 'p', 'e']
        else:
            eqn_type = 'burgers'
            ylabels = ['u']

        override_history = (state_history is not None)

        error_or_state = "error" if use_error else "state"

        if location is None and timestep is None:
            if not override_history:
                state_history = self.grid.get_full().copy()
                if use_error:
                    solution_state_history = self.solution.get_full().copy()
                num_steps = self.steps
            else:
                num_steps = len(state_history[0])
            if suffix is None:
                suffix = ("_step{:0" + str(self._step_precision) + "}").format(num_steps)
        else:
            if use_error and not override_history and not self.solution.is_recording_state():
                raise Exception("Can't save error if solution is not recording state.")

            if not override_history:
                state_history = np.array(self.state_history)
                if use_error:
                    solution_state_history = np.array(self.solution.get_state_history())

            if location is not None:
                if not override_history or history_includes_ghost:
                    location = self.ng + location
                state_history = state_history[:, :, location]
                if use_error:
                    solution_state_history = solution_state_history[:, :, location]
                actual_location = self.grid.x[location]
                if suffix is None:
                    suffix = ("_step{:0" + str(self._cell_index_precision) + "}").format(location)
            else:
                state_history = state_history[:, timestep, :]
                if use_error:
                    solution_state_history = solution_state_history[:, timestep, :]
                if suffix is None:
                    suffix = ("_step{:0" + str(self._step_precision) + "}").format(timestep)

        if eqn_type == 'euler':
            state_history = self.euler_state_conversion(state_history)
            if use_error:
                solution_state_history = self.euler_state_conversion(solution_state_history)

        if use_error:
            state_history = np.abs(solution_state_history - state_history)

        if location is None:
            if show_ghost:
                x_values = self.grid.x
            else:
                x_values = self.grid.real_x
                if not override_history or history_includes_ghost:
                    state_history = state_history[:, self.ng:-self.ng]
        else:
            if self.C is None:
                x_values = self.fixed_step * np.arange(len(state_history))
            else:
                # Need to record time values with variable timesteps.
                x_values = np.arange(len(state_history))

        log_dir = logger.get_dir()
        filename = "{}_{}{}.csv".format(eqn_type, error_or_state, suffix)
        filename = os.path.join(log_dir, filename)
        csv_file = open(filename, 'w')

        # Write column headers.
        if location is None:
            csv_file.write('x')
        elif self.C is None:
            csv_file.write('t')
        else:
            csv_file.write('timestep')
        for component in ylabels:
            if use_error:
                csv_file.write(f",error_{component}")
            else:
                csv_file.write(f",{component}")
        csv_file.write('\n')

        # Write data.
        for x_index, x in enumerate(x_values):
            csv_file.write(str(x))
            for component_index in range(len(state_history)):
                csv_file.write(f",{state_history[component_index, x_index]}")
            csv_file.write('\n')
        csv_file.close()
        if not silent:
            print('Saved data to ' + filename + '.')
        return filename

    def plot_state(self,
            timestep=None, location=None,
            plot_error=False,
            suffix=None, title=None,
            fixed_axes=False, no_borders=False, show_ghost=True,
            state_history=None, solution_state_history=None, weno_state_history=None,
            history_includes_ghost=True,
            silent=False):
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
            Index of location at which to plot the state.
        plot_error : bool
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
        silent : bool
            If False, print a message saying where the data was saved to.
        """

        assert (timestep is None or location is None), "Can't plot state at both a timestep and a location."

        if 'Euler' in str(self):
            eqn_type = 'euler'
            ylabels = ['rho', 'u', 'p', 'e']
        else:
            eqn_type = 'burgers'
            ylabels = ['u']

        self.set_colors()

        override_history = (state_history is not None)

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
                num_steps = len(state_history[0])
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
                state_history = state_history[:, :, location]
                if solution_state_history is not None:
                    solution_state_history = solution_state_history[:, :, location]
                if weno_state_history is not None:
                    weno_state_history = weno_state_history[:, :, location]

                actual_location = self.grid.x[location]
                if title is None:
                    title = "{} at x = {:.4f} (i = {})".format(error_or_state, actual_location, location)
                if suffix is None:
                    suffix = ("_step{:0" + str(self._cell_index_precision) + "}").format(location)
            else:
                state_history = state_history[:, timestep, :]
                if solution_state_history is not None:
                    solution_state_history = solution_state_history[:, timestep, :]
                if weno_state_history is not None:
                    weno_state_history = weno_state_history[:, timestep, :]

                if title is None:
                    actual_time = self.timestep_history[timestep]
                    title = ("{} at t = {:.4f} (step {:0" + str(self._step_precision) + "d})").format(
                            error_or_state, actual_time, timestep)
                if suffix is None:
                    suffix = ("_step{:0" + str(self._step_precision) + "}").format(timestep)

        if plot_error:
            if solution_state_history is None:
                raise Exception("Cannot plot error if solution state is not available.")

            state_history = np.abs(solution_state_history - state_history)

            if weno_state_history is not None:
                weno_state_history = np.abs(solution_state_history - weno_state_history)

            solution_state_history = None

        if eqn_type == 'euler':
            state_history = self.euler_state_conversion(state_history)
            if solution_state_history is not None:
                solution_state_history = self.euler_state_conversion(solution_state_history)
            if weno_state_history is not None:
                weno_state_history = self.euler_state_conversion(weno_state_history)

        vec_len = len(state_history)
        fig, ax = plt.subplots(nrows=vec_len, ncols=1, figsize=[6.4, 4.8 * vec_len], dpi=100)
        try:
            len(ax)
        except TypeError:
            ax = [ax]  # a hacky way to make subplots work with only one subplot

        if location is None:
            if show_ghost and (not override_history or history_includes_ghost):
                # The ghost arrays slice off one real point so the line connects to the real points.
                num_ghost_points = self.ng + 1

                ghost_x_left = self.grid.x[:num_ghost_points]
                ghost_x_right = self.grid.x[-num_ghost_points:]

                for i in range(vec_len):
                    ax[i].plot(ghost_x_left, state_history[i, :num_ghost_points],
                               ls='-', color=self.agent_ghost_color)
                    ax[i].plot(ghost_x_right, state_history[i, -num_ghost_points:],
                               ls='-', color=self.agent_ghost_color)

                    if solution_state_history is not None:
                        ax[i].plot(ghost_x_left, solution_state_history[i, :num_ghost_points],
                                   ls='-', color=self.true_ghost_color)
                        ax[i].plot(ghost_x_right, solution_state_history[i, -num_ghost_points:],
                                   ls='-', color=self.true_ghost_color)

                    if weno_state_history is not None:
                        ax[i].plot(ghost_x_left, weno_state_history[i, :num_ghost_points],
                                   ls='-', color=self.weno_ghost_color)
                        ax[i].plot(ghost_x_right, weno_state_history[i, -num_ghost_points:],
                                   ls='-', color=self.weno_ghost_color)

            state_history = state_history[:, self.ng:-self.ng]
            if solution_state_history is not None:
                solution_state_history = solution_state_history[:, self.ng:-self.ng]
            if weno_state_history is not None:
                weno_state_history = weno_state_history[:, self.ng:-self.ng]

        if location is None:
            x_values = self.grid.x[self.ng:-self.ng]
        else:
            if self.C is None:
                x_values = self.fixed_step * np.arange(len(state_history))
            else:
                # Need to record time values with variable timesteps.
                x_values = np.arange(len(state_history))

        if solution_state_history is not None:
            for i in range(vec_len):
                ax[i].plot(x_values, solution_state_history[i], ls='-', color=self.true_color,
                           label=self.solution_label)
        if weno_state_history is not None:
            if plot_error:
                weno_label = f"|{self.weno_solution_label} - {self.solution_label}|"
            else:
                weno_label = self.weno_solution_label
            for i in range(vec_len):
                ax[i].plot(x_values, weno_state_history[i], ls='-', color=self.weno_color,
                           label=weno_label)

        # Plot the agent line last so it is on the top.
        if plot_error:
            agent_label = f"|RL - {self.solution_label}|"
        else:
            agent_label = "RL"
        for i in range(vec_len):
            ax[i].plot(x_values, state_history[i], ls='-', color=self.agent_color, label=agent_label)
            ax[i].legend()
            if no_borders:
                ax[i].set_xmargin(0.0)
            ax[i].set_xlabel('x')
            ax[i].set_ylabel(f'{ylabels[i]}')

        ax[0].set_title(title)

        # Restrict y-axis if plotting abs error.
        # Can't have negative, cut off extreme errors.
        if plot_error:
            extreme_cutoff = 50.0
            for i in range(vec_len):
                max_not_extreme = np.max(state_history[i][state_history[i] < extreme_cutoff])
                if weno_state_history is not None:
                    max_weno = np.max(weno_state_history[i][weno_state_history[i] <
                        extreme_cutoff])
                    max_not_extreme = max(max_weno, max_not_extreme)
                ymax = max_not_extreme*1.05 if max_not_extreme > 0.0 else 0.01
                ax[i].set_ylim((0.0, ymax))
                if matplotlib.__version__ == '3.2.2':
                    ax[i].set_yscale('symlog', linthreshy=1e-9, subsy=range(2,10))
                else:
                    ax[i].set_yscale('symlog', linthresh=1e-9, subs=range(2,10))

        if fixed_axes:
            if not plot_error:
                if self._state_axes is None:
                    self._state_axes = (ax[0].get_xlim(), ax[0].get_ylim())
                else:
                    xlim, ylim = self._state_axes
                    for i in range(vec_len):
                        ax[i].set_xlim(xlim)
                        ax[i].set_ylim(ylim)
            else:
                pass # Fixed axes don't look as good on error plots.
                #if self._error_axes is None:
                    #self._error_axes = (ax[0].get_xlim(), ax[0].get_ylim())
                #else:
                    #xlim, ylim = self._error_axes
                    #for i in range(vec_len):
                        #ax[i].set_xlim(xlim)
                        #ax[i].set_ylim(ylim)

        fig.tight_layout()

        log_dir = logger.get_dir()
        filename = "{}_{}{}.png".format(eqn_type, error_or_state, suffix)
        filename = os.path.join(log_dir, filename)
        plt.savefig(filename)
        if not silent:
            print('Saved plot to ' + filename + '.')

        plt.close(fig)
        return filename

    def plot_state_evolution(self,
            num_states=10, full_true=False, no_true=False, only_true=False, plot_error=False, plot_weno=False,
            suffix="", title=None,
            state_history=None, solution_state_history=None, weno_state_history=None,
            silent=False, paper_mode=False):
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
        only_true : bool
            ONLY plot the true solution, not the RL solution. Sets full_true=True.
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
        silent : bool
            If False, print a message saying where the data was saved to.
        """
        self.set_colors()

        if only_true:
            full_true = True
            no_true = False

        if paper_mode:
            original_font_size = plt.rcParams['font.size']
            plt.rcParams.update({'font.size':15})

        if 'Euler' in str(self):
            eqn_type = 'euler'
            ylabels = ['rho', 'u', 'p', 'e']
        else:
            eqn_type = 'burgers'
            ylabels = ['u']

        override_history = (state_history is not None)

        vec_len = self.grid.space.shape[0]
        fig, ax = plt.subplots(nrows=vec_len, ncols=1, figsize=[6.4, 4.8 * vec_len], dpi=100)
        try:
            len(ax)
        except TypeError:
            ax = [ax]  # a hacky way to make subplots work with only one subplot

        x_values = self.grid.x[self.ng:-self.ng]

        if not override_history:
            state_history = np.array(self.state_history)[:, :, self.ng:-self.ng]
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
        weno_override = False
        if not plot_error and not no_true:
            if not override_history:
                # Decide whether to use solution or weno_solution, or both.
                # Would it make sense to have this decision be external in the __init__ method?
                if not isinstance(self.solution, OneStepSolution) and self.solution.is_recording_state():
                    solution_state_history = np.array(self.solution.get_state_history())[:, :, self.ng:-self.ng]
                    if (plot_weno and self.weno_solution is not None and
                                    self.weno_solution.is_recording_state()):
                        weno_state_history = np.array(
                                self.weno_solution.get_state_history())[:, :, self.ng:-self.ng]

                elif self.weno_solution is not None and self.weno_solution.is_recording_state():
                    solution_state_history = np.array(
                                self.weno_solution.get_state_history())[:, :, self.ng:-self.ng]
                    weno_override = True

            if solution_state_history is not None:
                assert len(state_history) == len(solution_state_history), "History mismatch."

                true_color = self.true_color
                if weno_override:
                    true_color = self.weno_color
                if full_true:
                    true_rgb = matplotlib.colors.to_rgb(true_color)
                    true_color_sequence = color_sequence(start_rgb, true_rgb, num_states)
                    sliced_solution_history = solution_state_history[slice_indexes]

                    for i in range(vec_len):
                        for state_values, color in zip(sliced_solution_history[1:-1], true_color_sequence[1:-1]):
                            ax[i].plot(x_values, state_values[i], ls='-', linewidth=1, color=color)
                        true = ax[i].plot(x_values, solution_state_history[-1, i], ls='-', linewidth=1, color=true_rgb)
                else:
                    for i in range(vec_len):
                        true = ax[i].plot(x_values, solution_state_history[-1, i],  ls='-', linewidth=4, color=true_color)

                if weno_state_history is not None:
                    assert len(state_history) == len(weno_state_history), "History mismatch."
                    if full_true:
                        weno_rgb = matplotlib.colors.to_rgb(self.weno_color)
                        weno_color_sequence = color_sequence(start_rgb, weno_rgb, num_states)
                        sliced_solution_history = weno_state_history[slice_indexes]

                        for i in range(vec_len):
                            for state_values, color in zip(sliced_solution_history[1:-1], weno_color_sequence[1:-1]):
                                ax[i].plot(x_values, state_values[i], ls='-', linewidth=1, color=color)
                            weno = ax[i].plot(x_values, weno_state_history[-1, i],
                                              ls='-', linewidth=1, color=self.weno_color)
                    else:
                        for i in range(vec_len):
                            weno = ax[i].plot(x_values, weno_state_history[-1, i],
                                              ls='-', linewidth=4, color=self.weno_color)

        if plot_error:
            if not override_history:
                if not self.solution.is_recording_state():
                    raise Exception("Cannot plot evolution of error if solution state is not available.")
                solution_state_history = np.array(self.solution.get_state_history())[:, :, self.ng:-self.ng]
                state_history = np.abs(solution_state_history - state_history)


        if not plot_error:
            for i in range(vec_len):
                init = ax[i].plot(x_values, state_history[0, i], ls='--', color=start_rgb)

        if full_true and not plot_error and not no_true:
            agent_rgb = matplotlib.colors.to_rgb(self.agent_color)
        else:
            # Use black if no color contrast is needed.
            agent_rgb = (0.0, 0.0, 0.0)
        agent_color_sequence = color_sequence(start_rgb, agent_rgb, num_states)
        sliced_history = state_history[slice_indexes]
        if not only_true:
            for i in range(vec_len):
                for state_values, color in zip(sliced_history[1:-1], agent_color_sequence[1:-1]):
                    ax[i].plot(x_values, state_values[i, :], ls='-', color=color)
                agent = ax[i].plot(x_values, state_history[-1, i], ls='-', color=agent_rgb)

        if plot_error:
            plots = [agent[0]]
            labels = ["|error|"]
        else:
            plots = [init[0]]
            labels = ["init"]
            if not only_true:
                plots.append(agent[0])
                labels.append("RL")
            if true is not None:
                plots.append(true[0])
                if weno_override:
                    labels.append(self.weno_solution_label)
                else:
                    if only_true:
                        labels.append("solution")
                    else:
                        labels.append(self.solution_label)
            if weno is not None:
                plots.append(weno[0])
                labels.append(self.weno_solution_label)

        if title is not None:
            ax[0].set_title(title)

        for i in range(vec_len):
            ax[i].legend(plots, labels)
            ax[i].set_xmargin(0.0)
            ax[i].set_xlabel('x')
            ax[i].set_ylabel(f'{ylabels[i]}')

        # Restrict y-axis if plotting abs error.
        # Can't have negative, cut off extreme errors.
        if plot_error:
            extreme_cutoff = 3.0
            for i in range(vec_len):
                max_not_extreme = np.max(state_history[i][state_history[i] < extreme_cutoff])
                ymax = max_not_extreme*1.05 if max_not_extreme > 0.0 else 0.01
                ax[i].set_ylim((0.0, ymax))
                if matplotlib.__version__ == '3.2.2':
                    ax[i].set_yscale('symlog', linthreshy=1e-9, subsy=range(2,10))
                else:
                    ax[i].set_yscale('symlog', linthresh=1e-9, subs=range(2,10))


        fig.tight_layout()

        log_dir = logger.get_dir()
        error_or_state = "error" if plot_error else "state"
        filename = os.path.join(log_dir,
                "{}_evolution_{}{}.png".format(eqn_type, error_or_state, suffix))
        plt.savefig(filename)
        if not silent:
            print('Saved plot to ' + filename + '.')

        plt.close(fig)
        if paper_mode:
            plt.rcParams.update({'font.size': original_font_size})
        return filename

    def save_action(self, timestep=None, location=None, suffix=None):
        """
        Save action data to a CSV file.

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
      
        """
        self.set_colors()

        assert (timestep is None or location is None), "Can't save action at both a timestep and a location."

        action_history = np.array(self.action_history)

        if 'Euler' in str(self):
            action_dimensions = np.prod(list(self.action_space.shape)[2:])
            vector_dimensions = self.action_space.shape[0]
            eqn_type = 'euler'
            ylabels = ['rho', 'u', 'p', 'e']
        else:  # Burgers
            action_dimensions = np.prod(list(self.action_space.shape)[1:])
            vector_dimensions = 1
            action_history = np.expand_dims(action_history, 1)
            eqn_type = 'burgers'
            ylabels = ['u']

        new_shape = (action_history.shape[0], action_history.shape[1], action_history.shape[2], action_dimensions)
        action_history = action_history.reshape(new_shape)

        # If plotting actions at a timestep, need to transpose location to the last dimension.
        # If plotting actions at a location, need to transpose time to the last dimension.
        if location is not None:
            action_history = action_history[:, :, location, :].transpose()

            actual_location = self.grid.x[location] - self.grid.dx/2
        else:
            if timestep is None:
                timestep = len(action_history) - 1
            action_history = action_history[timestep, :, :, :].transpose()

        if location is None:
            x_values = self.grid.inter_x[self.ng:-self.ng]
        else:
            if self.C is None:
                x_values = self.fixed_step * np.arange(len(state_history))
            else:
                # Need to record time values with variable timesteps.
                x_values = np.arange(len(state_history))

        log_dir = logger.get_dir()
        if suffix is None:
            if location is not None:
                suffix = ("_step{:0" + str(self._cell_index_precision) + "}").format(location)
            else:
                suffix = ("_step{:0" + str(self._step_precision) + "}").format(timestep)
        filename = f"action{suffix}.csv"
        filename = os.path.join(log_dir, filename)
        csv_file = open(filename, 'w')

        # Write column headers.
        if location is None:
            csv_file.write('x')
        elif self.C is None:
            csv_file.write('t')
        else:
            csv_file.write('timestep')
        if self._action_labels is None:
            action_labels = [f"$\omega_{i}$" for i in range(action_dimensions)]
        else:
            action_labels = self._action_labels
        for vector_part_label in ylabels:
            for action_part_label in self._action_labels:
                sep_char = '@' # In case labels have TeX in them, the @ is rarely used.
                assert sep_char not in vector_part_label and sep_char not in action_part_label
                action_label = f"{vector_part_label}{sep_char}{action_part_label}"
                csv_file.write("," + action_label)
        csv_file.write('\n')

        # Write data.
        for x_index, x in enumerate(x_values):
            csv_file.write(str(x))
            for vector_index in range(vector_dimensions):
                for action_index in range(action_dimensions):
                    csv_file.write("," + str(action_history[action_index, x_index, vector_index]))
            csv_file.write('\n')
        csv_file.close()
        print('Saved data to ' + filename + '.')
        return filename

    def plot_action(self, timestep=None, location=None, suffix=None, title=None,
                    fixed_axes=False, no_borders=False, **kwargs):
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
        self.set_colors()

        assert (timestep is None or location is None), "Can't plot action at both a timestep and a location."

        action_history = np.array(self.action_history)

        if 'Euler' in str(self):
            action_dimensions = np.prod(list(self.action_space.shape)[2:])
            vector_dimensions = self.action_space.shape[0]
            eqn_type = 'euler'
            ylabels = ['rho', 'u', 'p', 'e']
        else:  # Burgers
            action_dimensions = np.prod(list(self.action_space.shape)[1:])
            vector_dimensions = 1
            action_history = np.expand_dims(action_history, 1)
            eqn_type = 'burgers'
            ylabels = ['u']

        vertical_size = 5 * vector_dimensions
        horizontal_size = 4 * action_dimensions
        fig, axes = plt.subplots(vector_dimensions, action_dimensions, sharex=True, sharey=True,
                                 figsize=(horizontal_size, vertical_size))

        try:
            len(axes[0])
        except TypeError:
            axes = np.expand_dims(axes, 0)  # a hacky way to make subplots work with only one subplot

        if self.solution.is_recording_actions():
            weno_action_history = np.array(self.solution.get_action_history())
            weno_color = self.true_color
            assert(action_history.shape == weno_action_history.shape)
        elif self.weno_solution is not None and self.weno_solution.is_recording_actions():
            weno_action_history = np.array(self.weno_solution.get_action_history())
            weno_color = self.weno_color
            assert action_history.shape == weno_action_history.shape, \
                    f"{action_history.shape} != {weno_action_history.shape}"
        else:
            weno_action_history = None

        new_shape = (action_history.shape[0], action_history.shape[1], action_history.shape[2], action_dimensions)
        action_history = action_history.reshape(new_shape)
        if weno_action_history is not None:
            weno_action_history = weno_action_history.reshape(new_shape)

        # If plotting actions at a timestep, need to transpose location to the last dimension.
        # If plotting actions at a location, need to transpose time to the last dimension.
        if location is not None:
            action_history = action_history[:, :, location, :].transpose()
            if weno_action_history is not None:
                weno_action_history = weno_action_history[:, :, location, :].transpose()

            actual_location = self.grid.x[location] - self.grid.dx/2
            if title is None:
                fig.suptitle("actions at x = {:.4} (i = {} - 1/2)".format(actual_location, location))
            else:
                fig.suptitle(title)
        else:
            if timestep is None:
                timestep = len(action_history) - 1
            action_history = action_history[timestep, :, :, :].transpose()
            if weno_action_history is not None:
                weno_action_history = weno_action_history[timestep, :, :, :].transpose()

            if title is None:
                actual_time = self.timestep_history[timestep]
                fig.suptitle("actions at t = {:.4} (step {})".format(actual_time, timestep))
            else:
                fig.suptitle(title)

        if location is None:
            x_values = self.grid.inter_x[self.ng:-self.ng]
        else:
            if self.C is None:
                x_values = self.fixed_step * np.arange(len(state_history))
            else:
                # Need to record time values with variable timesteps.
                x_values = np.arange(len(state_history))

        for i in range(vector_dimensions):
            for j in range(action_dimensions):
                ax = axes[i][j]

                if weno_action_history is not None:
                    ax.plot(x_values, weno_action_history[j, :, i], c=weno_color, linestyle='-', label="WENO")
                ax.plot(x_values, action_history[j, :, i], c=self.agent_color, linestyle='-', label="RL")

                if no_borders:
                    ax.set_xmargin(0.0)

                if fixed_axes:
                   if self._action_axes is None:
                       self._action_axes = (ax.get_xlim(), ax.get_ylim())
                   else:
                       xlim, ylim = self._action_axes
                       ax.set_xlim(xlim)
                       ax.set_ylim(ylim)

                ax.legend()

        if self._action_labels is not None:
            for id, ax in enumerate(axes[0, :]):
                ax.set_title(self._action_labels[id])

        for id, ax in enumerate(axes[:, 0]):
            ax.set_ylabel(f"{ylabels[id]} actions")

        fig.tight_layout()

        log_dir = logger.get_dir()
        if suffix is None:
            if location is not None:
                suffix = ("_step{:0" + str(self._cell_index_precision) + "}").format(location)
            else:
                suffix = ("_step{:0" + str(self._step_precision) + "}").format(timestep)
        filename = 'action' + suffix + '.png'

        filename = os.path.join(log_dir, filename)
        plt.savefig(filename)
        print('Saved plot to ' + filename + '.')

        plt.close(fig)
        return filename

class Plottable2DEnv(AbstractPDEEnv):
    """
    Extension of PDE environment with plotting functions for scalars in 2 dimensions.

    Note that this is an abstract class - you can't declare a Plottable2DEnv.
    A subclass should extend this (and possibly other classes), then implement the requirements
    of subclasses of AbstractPDEEnv.
    """

    metadata = {'render.modes': ['file']}

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self._state_axes = None

        # Pillow is very likely to be available, but check for others just in case it's not.
        if "pillow" in animation.writers.list():
            self.animation_writer = animation.writers["pillow"]
            self.animation_extension = ".gif"
        elif "ffmpeg" in animation.writers.list():
            self.animation_writer = animation.writers["ffmpeg"]
            self.animation_extension = ".mp4"
        elif "imagemagick" in animation.writers.list():
            self.animation_writer = animation.writers["imagemagick"]
            self.animation_extension = ".gif"
        else:
            self.animation_writer = None
            self.animation_extension = None

    def render(self, mode='file', **kwargs):
        if mode is None:
            return
        if "file" in mode:
            return self.plot_state(**kwargs)
        else:
            raise Exception("Plottable1DEnv: \"{}\" render mode not recognized".format(mode))

    def _plot_state(self,
            axes,
            plot_error,
            title,
            fixed_axes, no_borders,
            state,
            state_includes_ghost):

        ax = axes

        if state_includes_ghost:
            x_values = self.grid.x
            y_values = self.grid.y
        else:
            x_values = self.grid.real_x
            y_values = self.grid.real_y

        #fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
        x, y = np.meshgrid(x_values, y_values, indexing='ij')
        z = state
        #if plot_error:
            #z = np.log10(z)
        surface = ax.plot_surface(x, y, z, cmap=cm.viridis,
                linewidth=0, antialiased=False)

        ax.set_title(title)

        if no_borders:
            ax.set_xmargin(0.0)
            ax.set_ymargin(0.0)

        # plot_error == True means state is actually abs(error), not the state.
        # Restrict z-axis if plotting error.
        # Can't have negative, cut off extreme errors.
        if plot_error:
            extreme_cutoff = 3.0
            max_not_extreme = np.max(state[state < extreme_cutoff])
            zmax = max_not_extreme*1.05 if max_not_extreme > 0.0 else 0.01
            ax.set_zlim((0.0, zmax))

        if fixed_axes:
            #TODO Keep the colorbar fixed as well.
            if self._state_axes is None:
                self._state_axes = (ax.get_xlim(), ax.get_ylim(), ax.get_zlim())
            else:
                xlim, ylim, zlim = self._state_axes
                ax.set_xlim(xlim)
                ax.set_ylim(ylim)
                ax.set_zlim(zlim)

        #if plot_error:
            ##stackoverlow.com/questions/3909794
            #def log_tick_formatter(val, pos=None):
                #return f"$10^{{{int(val)}}}$"
            #ax.zaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(log_tick_formatter))
            #ax.zaxis.set_major_locator(matplotlib.ticker.MaxNLocator(integer=True))
        #else:
        ax.zaxis.set_major_formatter(matplotlib.ticker.StrMethodFormatter("{x:.4g}"))
        ax.zaxis.set_major_locator(LinearLocator(10))
        #fig.colorbar(surface, shrink=0.5, aspect=5)

        # Indicate ghost cells with a rectangle around the real portion.
        #
        # Turns out, matplotlib isn't terribly well suited to drawing lines on existing 3D
        # surfaces. It also doesn't help that the boundary is almost indistinguishably close
        # to the edge. For now, show_ghost defaults to False, and setting it to true will
        # include the ghost cells but not indicate them in any way.
        """
        if history_includes_ghost:
            boundary_kwargs = {'color': 'k', 'linestyle': '-', 'linewidth': 1.5}
            lift = 0.01 # Keep the line above the surface so it's visible.
            x_num, y_num = self.grid.num_cells
            x_ghost, y_ghost = self.grid.num_ghosts
            x_min, y_min = self.grid.min_value
            x_max, y_max = self.grid.max_value

            left_y = self.grid.inter_y[y_ghost:-y_ghost]
            left_x = np.full_like(left_y, x_min)
            left_cells = state_history[x_ghost-1:x_ghost+1, y_ghost-1:-(y_ghost-1)]
            # This is a bit tricky because we don't HAVE the points on the boundaries - we have
            # cell-centered values. Average them to get the actual points we're looking for.
            left_z = (left_cells[:-1, :-1] + left_cells[:-1, 1:] 
                    + left_cells[1:, :-1] + left_cells[1:, 1:]) / 4 + lift
            left_z = left_z.squeeze(0)
            left_line = ax.plot(left_x, left_y, left_z, **boundary_kwargs)

            bottom_x = self.grid.inter_x[x_ghost:-x_ghost]
            bottom_y = np.full_like(bottom_x, y_min)
            bottom_cells = state_history[x_ghost-1:-(x_ghost-1), y_ghost-1:y_ghost+1]
            bottom_z = (bottom_cells[:-1, :-1] + bottom_cells[:-1, 1:] 
                    + bottom_cells[1:, :-1] + bottom_cells[1:, 1:]) / 4 + lift
            bottom_z = bottom_z.squeeze(1)
            bottom_line = ax.plot(bottom_x, bottom_y, bottom_z, **boundary_kwargs)

            right_y = left_y
            right_x = np.full_like(right_y, x_max)
            right_cells = state_history[-(x_ghost+1):-(x_ghost-1), y_ghost-1:-(y_ghost-1)]
            right_z = (right_cells[:-1, :-1] + right_cells[:-1, 1:] 
                    + right_cells[1:, :-1] + right_cells[1:, 1:]) / 4 + lift
            right_z = right_z.squeeze(0)
            right_line = ax.plot(right_x, right_y, right_z, **boundary_kwargs)

            top_x = bottom_x
            top_y = np.full_like(top_x, y_max)
            top_cells = state_history[x_ghost-1:-(x_ghost-1), -(y_ghost+1):-(y_ghost-1)]
            top_z = (top_cells[:-1, :-1] + top_cells[:-1, 1:] 
                    + top_cells[1:, :-1] + top_cells[1:, 1:]) / 4 + lift
            top_z = top_z.squeeze(1)
            top_line = ax.plot(top_x, top_y, top_z, **boundary_kwargs)
        """

        #return fig

    def save_state(self, *args, **kwargs):
        print("2D CSV save function not yet implemented.")
        return "ERROR_NOT_SAVED"

    def plot_state(self,
            timestep=None,
            plot_error=False,
            suffix=None, title=None,
            fixed_axes=False, no_borders=False,
            show_ghost=False,
            state_history=None, solution_state_history=None,
            history_includes_ghost=True):

        override_history = (state_history is not None)
        if override_history and not history_includes_ghost:
            show_ghost = False

        error_or_state = "error" if plot_error else "state"

        if timestep is None:
            if not override_history:
                state_history = self.grid.get_full().copy()
                if plot_error:
                    solution_state_history = self.solution.get_full().copy()

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
                state_history = np.array(self.state_history)[timestep, :]
                if plot_error:
                    assert self.solution.is_recording_state(), "Past solution not recorded."
                    solution_state_history = self.solution.get_state_history().copy()[timestep, :]

            if title is None:
                actual_time = self.timestep_history[timestep]
                title = ("{} at t = {:.4f} (step {:0" + str(self._step_precision) + "d})").format(
                        error_or_state, actual_time, timestep)
            if suffix is None:
                suffix = ("_step{:0" + str(self._step_precision) + "}").format(timestep)

        if plot_error:
            if solution_state_history is None:
                raise Exception("Cannot plot error if solution state is not available.")

            state_history = np.abs(solution_state_history - state_history)

        if not show_ghost:
            state_history = state_history[self.grid.real_slice]


        fig, ax = plt.subplots(subplot_kw={"projection": "3d"})

        #TODO Handle vectors in a better way than simply using the first dimension.
        state_history = state_history[0]

        self._plot_state(axes=ax, plot_error=plot_error, title=title,
                fixed_axes=fixed_axes, no_borders=no_borders,
                state=state_history,
                state_includes_ghost=(override_history and history_includes_ghost))

        fig.tight_layout()

        log_dir = logger.get_dir()
        filename = "burgers2d_{}{}.png".format(error_or_state, suffix)
        filename = os.path.join(log_dir, filename)
        fig.savefig(filename)
        print('Saved plot to ' + filename + '.')

        plt.close(fig)
        return filename

    def plot_state_evolution(self, plot_error=False,
            show_ghost=False, no_true=False,
            suffix="", title=None, num_frames=50,
            state_history=None, history_includes_ghost=False,
            silent=False):

        if self.animation_writer is None:
            raise Exception("No familiar image libraries avilable to render evolution." + 
                    " These are the available libraries: " + str(animation.writers.list()))

        override_history = (state_history is not None)

        if not override_history:
            state_history = np.array(self.state_history)

        if title is None:
            base_title = ""
        else:
            base_title = title

        # Indexes into the state history. There are num_frames+1 indices, where the first
        # is always 0 the last is always len(state_history)-1, and the rest are evenly
        # spaced between them.
        timesteps = (np.arange(num_frames+1)*(len(self.state_history)-1)/num_frames).astype(int)

        fig, ax = plt.subplots(subplot_kw={"projection": "3d"})

        def update_plot(timestep):

            if timestep == -1:
                return []

            state = state_history[timestep]

            actual_time = self.timestep_history[timestep]
            time_str = (" t = {:.4f} (step {:0" + str(self._step_precision) + "d})").format(
                            actual_time, timestep)
            title = base_title + time_str

            if not show_ghost and (not override_history or history_includes_ghost):
                state = state[self.grid.real_slice]

            state_includes_ghost = (show_ghost and (not override_history or
                history_includes_ghost))

            #TODO Handle vectors in a better way than simply using the first dimension.
            state = state[0]

            ax.cla()
            self._plot_state(axes=ax, plot_error=False, title=title,
                    fixed_axes=True, no_borders=True,
                    state=state,
                    state_includes_ghost=state_includes_ghost)

            # TODO also plot error (unless no_true == True)

            if timestep == 0:
                fig.tight_layout()

            return [ax]

        # In milliseconds.
        frame_interval = 100
        start_delay = 500
        end_delay = 500

        start_delay_frames = [-1] * int(start_delay / frame_interval)
        end_delay_frames = [-1] * int(end_delay / frame_interval)

        frames = [0,] + start_delay_frames + list(timesteps[1:]) + end_delay_frames

        ani = animation.FuncAnimation(fig, update_plot, frames, interval=frame_interval)
        # FuncAnimation has an argument "repeat_delay" which serves the same purpose as end_delay,
        # except repeat_delay doesn't work for saved animations.

        fps = 1000.0 / frame_interval
        writer = self.animation_writer(fps=fps)

        log_dir = logger.get_dir()
        filename = os.path.join(log_dir,
                "evolution{}{}".format(suffix, self.animation_extension))

        if not silent:
            print("Saving animation to {}".format(filename), end='')
        ani.save(filename, writer,
                progress_callback = lambda i, n: print('.', end='', flush=True))
        if not silent:
            print('Saved.')

        plt.close(fig)
        ani = None

        return filename

    def plot_action(self, *args, **kwargs):
        raise NotImplementedError("2D action plot not implemented."
                + " Not sure the right way to do that. Should we have a 3D plot for each weight?")
