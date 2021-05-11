import json
import numpy as np
from scipy.optimize import fixed_point
from scipy.optimize import brentq

import envs.weno_coefficients as weno_coefficients
from envs.grid import GridBase, Grid1d


class SolutionBase(GridBase):
    """
    Base class for Solutions.

    A Solution acts like a grid except that one can update to the next state
    automatically given a time delta and target time.

    A Solution can also optionally keep track of the state as it changes, updating
    the state history on reset and update.

    Subclasses should extend _update and _reset;
    update and reset here are lightweight wrappers of _update and _reset
    that record the state history.
    """
    def __init__(self, record_state=False, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.record_state = False
        self.state_history = []

    def update(self, dt, time):
        self._update(dt, time)
        if self.record_state:
            self.state_history.append(self.get_full().copy())

    def reset(self, params):
        self._reset(params)
        if self.record_state:
            self.state_history = [self.get_full().copy()]

    def _reset(self, params):
        raise NotImplementedError()
    def _update(self, dt, time):
        raise NotImplementedError()

    def is_recording_state(self):
        return self.record_state
    def set_record_state(self, record_state):
        self.record_state = record_state
        if not self.record_state:
            self.state_history = []
    def get_state_history(self):
        return self.state_history

    def is_recording_actions(self):
        # Override this if you have get_action_history implemented.
        return False


class PreciseWENOSolution(SolutionBase):
    #TODO: should also calculate precise WENO with smaller timesteps.

    def __init__(self, nx, ng, xmin, xmax,
                 precise_order, precise_scale, init_type, boundary, flux_function,
                 eps=0.0, source=None,
                 record_state=False, record_actions=None):
        super().__init__(nx=nx, ng=ng, xmin=xmin, xmax=xmax)

        assert (precise_scale % 2 == 1), "Precise scale must be odd for easier downsampling."

        self.precise_scale = precise_scale

        self.precise_nx = precise_scale * nx
        if precise_order + 1 > precise_scale * ng:
            self.precise_ng = precise_order + 1
            self.extra_ghosts = self.precise_ng - (precise_scale * ng)
        else:
            self.precise_ng = precise_scale * ng
            self.extra_ghosts = 0
        self.precise_grid = Grid1d(xmin=xmin, xmax=xmax, nx=self.precise_nx, ng=self.precise_ng,
                                   boundary=boundary, init_type=init_type)

        self.flux_function = flux_function
        self.eps = eps
        self.order = precise_order
        self.source = source

        self.record_actions = record_actions
        self.action_history = []

    def is_recording_actions(self):
        return (self.record_actions is not None)
    def set_record_actions(self, record_mode):
        self.record_actions = record_mode
    def get_action_history(self):
        return self.action_history

    def weno_stencils(self, q):
        """
        Compute WENO stencils

        Parameters
        ----------
        q : np array
          Scalar data to reconstruct.

        Returns
        -------
        stencils

        """
        order = self.order
        a = weno_coefficients.a_all[order]
        num_points = len(q) - 2 * order
        q_stencils = np.zeros((order, len(q)))
        for i in range(order, num_points + order):
            for k in range(order):
                for l in range(order):
                    q_stencils[k, i] += a[k, l] * q[i + k - l]

        return q_stencils

    def weno_weights(self, q):
        """
        Compute WENO weights

        Parameters
        ----------
        q : np array
          Scalar data to reconstruct.

        Returns
        -------
        stencil weights

        """
        order = self.order
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

    def weno_new(self, q):
        """
        Compute WENO reconstruction

        Parameters
        ----------
        q : numpy array
          Scalar data to reconstruct.

        Returns
        -------
        qL: numpy array
          Reconstructed data.

        """

        weights = self.weno_weights(q)
        q_stencils = self.weno_stencils(q)
        qL = np.zeros_like(q)
        num_points = len(q) - 2 * self.order
        for i in range(self.order, num_points + self.order):
            qL[i] = np.dot(weights[:, i], q_stencils[:, i])
        return qL, weights

    def rk_substep(self):

        # get the solution data
        g = self.precise_grid

        # comput flux at each point
        f = self.flux_function(g.u)

        # get maximum velocity
        alpha = np.max(abs(g.u))

        # Lax Friedrichs Flux Splitting
        fp = (f + alpha * g.u) / 2
        fm = (f - alpha * g.u) / 2

        fpr = g.scratch_array()
        fml = g.scratch_array()
        flux = g.scratch_array()

        # compute fluxes at the cell edges
        # compute f plus to the right
        fpr[1:], fp_weights = self.weno_new(fp[:-1])
        # compute f minus to the left
        # pass the data in reverse order
        fml[-1::-1], fm_weights = self.weno_new(fm[-1::-1])

        if self.record_actions is not None:
            action_weights = np.stack((fp_weights[:, self.ng-1:-(self.ng-1)], fm_weights[:, -(self.ng+1):self.ng-2:-1]))
            # resulting array is (fp, fm) X stencil X location
            # transpose to location X (fp, fm) X stencil
            action_weights = action_weights.transpose((2, 0, 1))

            if self.record_actions == "weno":
                self.action_history.append(action_weights)
            elif self.record_actions == "coef":
                # Same as in Standard WENO agent in agents.py.
                order = self.order
                a_mat = weno_coefficients.a_all[order]
                a_mat = np.flip(a_mat, axis=-1)
                combined_weights = a_mat[None, None, :, :] * action_weights[:, :, :, None]

                flux_weights = np.zeros((g.real_length() + 1, 2, self.order * 2 - 1))
                for sub_stencil_index in range(order):
                    flux_weights[:, :, sub_stencil_index:sub_stencil_index + order] += combined_weights[:, :, sub_stencil_index, :]
                self.action_history.append(flux_weights)
            else:
                raise Exception("Unrecognized action type: '{}'".format(self.record_actions))

        # compute flux from fpr and fml
        flux[1:-1] = fpr[1:-1] + fml[1:-1]
        rhs = g.scratch_array()

        if self.eps > 0.0:
            R = self.eps * self.lap()
            rhs[1:-1] = 1 / g.dx * (flux[1:-1] - flux[2:]) + R[1:-1]
        else:
            rhs[1:-1] = 1 / g.dx * (flux[1:-1] - flux[2:])

        if self.source is not None:
            rhs[1:-1] += self.source.get_full()[1:-1]

        return rhs

    def lap(self):
        """
        Returns the Laplacian of g.u.

        This calculation relies on ghost cells, so make sure they have been filled before calling this.
        """

        gr = self.precise_grid
        u = gr.u

        lapu = gr.scratch_array()

        ib = gr.ilo - 1
        ie = gr.ihi + 1

        lapu[ib:ie + 1] = (u[ib - 1:ie] - 2.0 * u[ib:ie + 1] + u[ib + 1:ie + 2]) / gr.dx ** 2

        return lapu

    def _update(self, dt, time):
        # Do Euler step, though rk_substep is separated so this can be converted to RK4.
        self.t = time
        euler_step = dt * self.rk_substep()

        self.precise_grid.u += euler_step
        self.precise_grid.update_boundary()

    def get_full(self):
        """ Downsample the precise solution to get coarse solution values. """

        grid = self.precise_grid.get_full()
        if self.extra_ghosts > 0:
            grid = grid[self.extra_ghosts:-self.extra_ghosts]

        # For each coarse cell, there are precise_scale precise cells, where the middle
        # cell corresponds to the same location as the coarse cell.
        middle = int(self.precise_scale / 2)
        return grid[middle::self.precise_scale]

    def get_real(self):
        return self.get_full()[self.ng:-self.ng]

    def _reset(self, init_params):
        self.precise_grid.reset(init_params)
        self.action_history = []

    def set(self, real_values):
        """ Force set the current grid. Will make the state/action history confusing. """
        self.precise_grid.set(real_values)

class MemoizedSolution(SolutionBase):
    """
    Decorator on solutions that memoizes their state to save computation.
    
    Useful for when the solution uses the same parameters every episode, or uses one of a set
    of the same parameters.
    Wastes memory for solutions that change every episode.
    """
    def __init__(self, solution):
        assert not isinstance(solution, OneStepSolution), ("Memoized solutions are not compatible"
        + " with one-step solutions. (Memoized solutions stay the same whereas one-step solutions"
        + " always change).")
        super().__init__(nx=solution.nx, ng=solution.ng, xmin=solution.xmin, xmax=solution.xmax,
                # The inner solution should record the state history, not this wrapper.
                record_state=False)

        self.inner_solution = solution
        self.inner_solution.set_record_state(True)

        self.master_state_dict = {}
        self.master_action_dict = {}
        self.params_str = None

        self.isSavedSolution = False
        self.time_index = -1

        self.dt = None
        self.MAX_MEMOS = 100

    # Forward method calls that aren't available here to inner solution.
    # If you're not familiar, __getattr__ is only called when the
    # attr wasn't found by usual means.
    def __getattr__(self, attr):
        return getattr(self.inner_solution, attr)

    def _update(self, dt, time):
        if self.isSavedSolution:
            self.time_index += 1
        else:
            if self.dt is None:
                self.dt = dt
            else:
                assert self.dt == dt, "Memoized solutions were not designed to work with variable timesteps!"
            self.inner_solution.update(dt, time)

    def is_recording_state(self):
        assert self.inner_solution.is_recording_state()
        return True

    def is_recording_actions(self):
        return self.inner_solution.is_recording_actions()

    def set_record_state(self, record_state):
        if not record_state:
            raise Exception("Cannot stop recording state with MemoizedSolution.")

    def get_state_history(self):
        if self.isSavedSolution:
            return self.master_state_dict[self.params_str][:self.time_index+1]
        else:
            return self.inner_solution.get_state_history()

    def get_action_history(self):
        if self.isSavedSolution:
            return self.master_action_dict[self.params_str][:self.time_index+1]
        else:
            return self.inner_solution.get_action_history()

    def get_full(self):
        if self.isSavedSolution:
            return self.master_state_dict[self.params_str][self.time_index]
        else:
            return self.inner_solution.get_full()

    def get_real(self):
        return self.get_full()[self.ng:-self.ng]

    def _reset(self, init_params):
        # If time_index is -1, then this is the first call to reset,
        # and we don't have a potential solution to save.
        if not self.isSavedSolution and self.time_index != -1:
            if len(self.master_state_dict) < self.MAX_MEMOS:
                state_history = self.inner_solution.get_state_history().copy()
                self.master_state_dict[self.params_str] = state_history
                action_history = self.inner_solution.get_action_history().copy()
                self.master_action_dict[self.params_str] = action_history
            else:
                print(("MemoizedSolution: maximum number ({}) of saved solutions reached!"
                        + " Check that no parameters come from a continuous range.")
                        .format(self.MAX_MEMOS))

        params_str = json.dumps(init_params, ensure_ascii=True, sort_keys=True)
        self.params_str = params_str
        if params_str in self.master_state_dict:
            self.isSavedSolution = True
            self.time_index = 0
        else:
            self.isSavedSolution = False
            self.inner_solution.reset(init_params)
            self.time_index = -2

class OneStepSolution(SolutionBase):
    """
    A OneStepSolution is only one step different from a different grid.

    It keeps a reference to another grid, then, when updating first sets the
    grid to the same state as that other grid.

    Using a solution like this can be used for a more reliable comparison,
    as there is only one step's worth of different actions.
    """
    def __init__(self, solution, current_grid):
        assert not isinstance(solution, MemoizedSolution), ("Memoized solutions are not compatible"
        + " with one-step solutions. (Memoized solutions stay the same whereas one-step solutions"
        + " always change).")
        assert not isinstance(solution, AnalyticalSolution), ("One-step solutions are not compatible"
        + " with analytical solutions. (Analytical solutions stay the same whereas one-step solutions"
        + " always change).")
        super().__init__(nx=solution.nx, ng=solution.ng, xmin=solution.xmin, xmax=solution.xmax,
                record_state=False)
        self.inner_solution = solution
        self.current_grid = current_grid

    def _reset(self, params):
        self.inner_solution.reset(params)

    def _update(self, dt, time):
        current_state = self.current_grid.get_real().copy()
        self.inner_solution.set(current_state)
        self.inner_solution.update(dt, time)

    def get_real(self):
        return self.inner_solution.get_real()
    def get_full(self):
        return self.inner_solution.get_full()

    def is_recording_actions(self):
        return self.inner_solution.is_recording_actions()

    # Forward method calls that aren't available here to inner solution.
    # If you're not familiar, __getattr__ is only called when the
    # attr wasn't found by usual means.
    def __getattr__(self, attr):
        return getattr(self.inner_solution, attr)

available_analytical_solutions = ["smooth_sine", "smooth_rare", "accelshock", "gaussian"]
#TODO account for xmin, xmax in case they're not 0 and 1.
class AnalyticalSolution(SolutionBase):
    def __init__(self, nx, ng, xmin, xmax, init_type="schedule"):
        super().__init__(nx=nx, ng=ng, xmin=xmin, xmax=xmax)

        if (not init_type in available_analytical_solutions 
                and not init_type in ['schedule', 'sample']):
            raise Exception("Invalid analytical type \"{}\", available options are {}.".format(init_type, available_analytical_solutions))

        self.grid = Grid1d(nx=nx, ng=ng, xmin=xmin, xmax=xmax, init_type=init_type)

    def _update(self, dt, time):
        # Assume that Grid1d.reset() is still setting the correct parameters.
        params = self.grid.init_params
        init_type = params['init_type']

        x_values = self.grid.real_x

        if init_type in ["smooth_sine", "smooth_rare"]:
            if init_type == "smooth_sine":
                iterate_func = lambda old_u, time: params['A'] * np.sin(2 * np.pi * (x_values - old_u * time))
            elif init_type == "smooth_rare":
                iterate_func = lambda old_u, time: params['A'] * np.tanh(params['k'] * (x_values - 0.5 - old_u*time))

            try:
                updated_u = fixed_point(iterate_func, self.grid.get_real(), args=(time,))
                self.grid.set(updated_u)
            except Exception:
                print("failed to converge")
                #TODO handle this better

        elif init_type == "accelshock":
            offset = params['offset']
            u_L = params['u_L']
            u_R = params['u_R']
            shock_location = (u_L/u_R + 1) * (1 - np.sqrt(1 + u_R*time)) + u_L*time + offset*np.sqrt(u_R*time+1)
            new_u = np.full_like(x_values, u_L)
            index = x_values > shock_location
            new_u[index] = (u_R*(x_values[index] - 1)) / (1 + u_R*time)
            self.grid.set(new_u)

        elif init_type == "gaussian":
            # Copied from
            # github.com/python-hydro/hydro_examples/blob/master/burgers/weno_burgers.py#burgers_sin_exact
            # (Yes, their implementation of the exact Gaussian solution is called
            # "burgers_sin_exact".)
            # I'm not really sure how it works.
            def initial_gaussian(x):
                return 1.0 + np.exp(-60.0*(x - 0.5)**2)
            def residual(x_at_t_0_guess, x_at_t):
                q = initial_gaussian(x_at_t_0_guess)
                return x_at_t_0_guess + q*time - x_at_t

            updated_u = [initial_gaussian(brentq(residual, -2, 2, args=(x,))) for x in x_values]
            self.grid.set(updated_u)


    def _reset(self, params):
        if 'init_type' in params and not params['init_type'] in available_analytical_solutions:
                raise Exception("Invalid analytical type \"{}\", available options are {}.".format(init_type, available_analytical_solutions))

        self.grid.reset(params)

    def get_full(self):
        return self.grid.get_full()
    def get_real(self):
        return self.grid.get_real()

    def set(self, real_values):
        raise Exception("An analytical solution cannot be set.")
