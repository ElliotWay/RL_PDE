import os

import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.ticker import LinearLocator
import matplotlib.animation as animation
import numpy as np
from stable_baselines import logger

from envs.abstract_scalar_env import AbstractScalarEnv
from envs.solutions import OneStepSolution

class Plottable1DEnv(AbstractScalarEnv):
    """
    Extension of PDE environment with plotting functions for 1D scalars.

    Can probably be extended to plotting vectors without too much reworking.
    Note that this is an abstract class - you can't declare a Plottable1DEnv.
    A subclass should extend this (and possibly other classes), then implement the requirements
    of subclasses of AbstractScalarEnv.
    """
    metadata = {'render.modes': ['file']}

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self._state_axes = None
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

    def plot_state(self,
            timestep=None, location=None,
            plot_error=False,
            suffix=None, title=None,
            fixed_axes=False, no_borders=False, show_ghost=True,
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
        """

        assert (timestep is None or location is None), "Can't plot state at both a timestep and a location."

        self.set_colors()

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

        # I haven't done enough testing with timestep != None or location != None. Is this a bug?
        # Ghosts should be fine with past steps; it's with a location over time where it doesn't
        # make sense. Should this be if location is None?
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

        # Similarly here. With a specific timestep we still want physical x values. Should this
        # also be if location is None?
        if timestep is None:
            x_values = self.grid.x[self.ng:-self.ng]
        else:
            if self.C is None:
                x_values = self.fixed_step * np.arange(len(state_history))
            else:
                # Need to record time values with variable timesteps.
                x_values = np.arange(len(state_history))

        if solution_state_history is not None:
            plt.plot(x_values, solution_state_history, ls='-', color=self.true_color,
                    label=self.solution_label)
        if weno_state_history is not None:
            plt.plot(x_values, weno_state_history, ls='-', color=self.weno_color,
                    label=self.weno_solution_label)

        # Plot this one last so it is on the top.
        agent_label = "RL"
        if plot_error:
            agent_label = "|error|"
        plt.plot(x_values, state_history, ls='-', color=self.agent_color, label=agent_label)

        plt.legend(loc="upper right")
        ax = plt.gca()

        ax.set_title(title)

        if no_borders:
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
        self.set_colors()

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
        weno_override = False
        if not plot_error and not no_true:
            if not override_history:
                # Decide whether to use solution or weno_solution, or both.
                # Would it make sense to have this decision be external in the __init__ method?
                if not isinstance(self.solution, OneStepSolution) and self.solution.is_recording_state():
                    solution_state_history = np.array(self.solution.get_state_history())[:, self.ng:-self.ng]
                    if (plot_weno and self.weno_solution is not None and
                                    self.weno_solution.is_recording_state()):
                        weno_state_history = np.array(
                                self.weno_solution.get_state_history())[:, self.ng:-self.ng]

                elif self.weno_solution is not None and self.weno_solution.is_recording_state():
                    solution_state_history = np.array(
                                self.weno_solution.get_state_history())[:, self.ng:-self.ng]
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
                    
                    for state_values, color in zip(sliced_solution_history[1:-1], true_color_sequence[1:-1]):
                        plt.plot(x_values, state_values, ls='-', linewidth=1, color=color)
                    true = plt.plot(x_values, solution_state_history[-1],
                                    ls='-', linewidth=1, color=true_rgb)
                else:
                    true = plt.plot(x_values, solution_state_history[-1], 
                                    ls='-', linewidth=4, color=true_color)

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
                if weno_override:
                    labels.append(self.weno_solution_label)
                else:
                    labels.append(self.solution_label)
            if weno is not None:
                plots.append(weno[0])
                labels.append(self.weno_solution_label)
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

        real_x = self.grid.inter_x[self.ng:-self.ng]

        for dim in range(action_dimensions):
            ax = axes[dim]

            if weno_action_history is not None:
                ax.plot(real_x, weno_action_history[dim, :], c=weno_color, linestyle='-', label="WENO")
            ax.plot(real_x, action_history[dim, :], c=self.agent_color, linestyle='-', label="RL")

            if no_borders:
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

class Plottable2DEnv(AbstractScalarEnv):
    """
    Extension of PDE environment with plotting functions for scalars in 2 dimensions.

    Note that this is an abstract class - you can't declare a Plottable2DEnv.
    A subclass should extend this (and possibly other classes), then implement the requirements
    of subclasses of AbstractScalarEnv.
    """

    metadata = {'render.modes': ['file']}

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self._state_axes = None

        # Pillow is very likely to be available, but check for others just in case it's not.
        if "pillow" in animation.writers:
            self.animation_writer = animation.writers["pillow"]
            self.animation_extension = ".gif"
        elif "ffmpeg" in animation.writers:
            self.animation_writer = animation.writers["ffmpeg"]
            self.animation_extension = ".mp4"
        elif "imagemagick" in animation.writers:
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
            ax.set_zlim((0.0, ymax))

        if fixed_axes:
            #TODO Keep the colorbar fixed as well.
            if self._state_axes is None:
                self._state_axes = (ax.get_xlim(), ax.get_ylim(), ax.get_zlim())
            else:
                xlim, ylim, zlim = self._state_axes
                ax.set_xlim(xlim)
                ax.set_ylim(ylim)
                ax.set_zlim(zlim)

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

        if not show_ghost:
            state_history = state_history[self.grid.real_slice]


        fig, ax = plt.subplots(subplot_kw={"projection": "3d"})

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
            state_history=None, history_includes_ghost=False):

        if self.animation_writer is None:
            raise Exception("No familiar image libraries avilable to render evolution." + 
                    " These are the available libraries: " + str(animation.writers.list()))

        override_history = (state_history is not None)
        if override_history:
            print("state history shape:", state_history.shape)
        print("original history shape:", np.array(self.state_history).shape)

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

            if self.C is None:
                actual_time = timestep * self.fixed_step
                time_str = (" t = {:.4f} (step {:0" + str(self._step_precision) + "d})").format(
                                actual_time, timestep)
            else:
                # TODO get time with variable timesteps?
                time_str = " step {:0" + str(self._step_precision) + "d}".format(timestep)
            title = base_title + time_str

            if not show_ghost and (not override_history or history_includes_ghost):
                state = state[self.grid.real_slice]

            state_includes_ghost = (show_ghost and (not override_history or
                history_includes_ghost))

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

        print("Saving animation to {}".format(filename), end='')
        ani.save(filename, writer,
                progress_callback = lambda i, n: print('.', end='', flush=True))
        print('Saved.')

        plt.close(fig)
        ani = None

        return filename

    def plot_action(self, *args, **kwargs):
        raise NotImplementedError("2D action plot not implemented."
                + " Not sure the right way to do that. Should we have a 3D plot for each weight?")
