import json
import numpy as np
from scipy.optimize import fixed_point
from scipy.optimize import brentq

import envs.weno_coefficients as weno_coefficients
from envs.grid import AbstractGrid
from envs.grid1d import Burgers1DGrid, Euler1DGrid


class SolutionBase(AbstractGrid):
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
    def __init__(self, *args, record_state=False, **kwargs):
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


class MemoizedSolution(SolutionBase):
    """
    Decorator on solutions that memoizes their state to save computation.
    
    Useful for when the solution uses the same parameters every episode, or uses one of a set
    of the same parameters.
    Wastes memory for solutions that change every episode.
    """
    def __init__(self, solution, ep_length, vec_len=1):
        assert not isinstance(solution, OneStepSolution), ("Memoized solutions are not compatible"
        + " with one-step solutions. (Memoized solutions stay the same whereas one-step solutions"
        + " always change).")
        super().__init__(solution.num_cells, solution.num_ghosts, solution.min_value,
                solution.max_value, vec_len,
                # The inner solution should record the state history, not this wrapper.
                record_state=False)

        self.correct_solution_history_length = ep_length + 1

        self.inner_solution = solution
        self.inner_solution.set_record_state(True)

        self.master_state_dict = {}
        self.master_action_dict = {}
        self.params_str = None

        self.isSavedSolution = False
        self.time_index = -1

        self.dt = None
        self.MAX_MEMOS = 500

    # Forward method calls that aren't available here to inner solution.
    # If you're not familiar, __getattr__ is only called when the
    # attr wasn't found by usual means.
    def __getattr__(self, attr):
        return getattr(self.inner_solution, attr)

    def _update(self, dt, time):
        if self.isSavedSolution:
            self.time_index += 1
            if self.time_index >= len(self.get_state_history()):
                raise Exception("MemoizedSolution: too many updates!"
                        + " The saved solution is not long enough to account for the number"
                        + " of update calls. This likely occured because you called update()"
                        + " more times than the expected episode length.")
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
        return self.get_full()[self.real_slice]

    def _reset(self, init_params):
        # If time_index is -1, then this is the first call to reset,
        # and we don't have a potential solution to save.
        if (not self.isSavedSolution and self.time_index != -1
                and not len(self.inner_solution.get_state_history())
                    < self.correct_solution_history_length):
            if len(self.master_state_dict) < self.MAX_MEMOS:
                state_history = self.inner_solution.get_state_history().copy()
                self.master_state_dict[self.params_str] = state_history
                if self.inner_solution.is_recording_actions():
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
    def __init__(self, solution, current_grid, vec_len=1):
        assert not isinstance(solution, MemoizedSolution), ("Memoized solutions are not compatible"
        + " with one-step solutions. (Memoized solutions stay the same whereas one-step solutions"
        + " always change).")
        assert not isinstance(solution, AnalyticalSolution), ("One-step solutions are not compatible"
        + " with analytical solutions. (Analytical solutions stay the same whereas one-step solutions"
        + " always change).")
        super().__init__(solution.num_cells, solution.num_ghosts, solution.min_value,
                solution.max_value, vec_len, record_state=False)
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

#TODO Handle ND analytical solutions. (Also non-Burgers solutions?)
available_analytical_solutions = ["smooth_sine", "smooth_rare", "accelshock", "gaussian"]
#TODO account for xmin, xmax in case they're not 0 and 1.
class AnalyticalSolution(SolutionBase):
    def __init__(self, nx, ng, xmin, xmax, vec_len=1, init_type="schedule"):
        super().__init__(nx, ng, xmin, xmax, vec_len=vec_len)

        if (not init_type in available_analytical_solutions
                and not init_type in ['schedule', 'sample']):
            raise Exception("Invalid analytical type \"{}\", available options are {}.".format(init_type, available_analytical_solutions))

        self.grid = Burgers1DGrid(nx, ng, xmin, xmax, init_type)

    def _update(self, dt, time):
        # Assume that Burgers1DGrid.reset() is still setting the correct parameters.
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


class RiemannSolution(SolutionBase):
    def __init__(self, nx, ng, xmin, xmax, vec_len=3, init_type="sod", gamma=1.4):
        super().__init__(nx, ng, xmin, xmax, vec_len)

        self.grid = Euler1DGrid(nx=nx, ng=ng, xmin=xmin, xmax=xmax, init_type=init_type)
        self.gamma = gamma

    def euler_state_conversion(self, state):
        rho = state[0]
        v = state[1] / state[0]
        p = (self.gamma - 1) * (state[2] - rho * v ** 2 / 2)
        return np.stack([rho, v, p])

    def u_hugoniot(self, p, side, shock=False):
        """define the Hugoniot curve, u(p).  If shock=True, we do a 2-shock
        solution"""

        if side == "left":
            state = self.left
            s = 1.0
        elif side == "right":
            state = self.right
            s = -1.0

        c = np.sqrt(self.gamma*state[2]/state[0])

        if shock:
            # shock
            beta = (self.gamma+1.0)/(self.gamma-1.0)
            u = state[1] + s*(2.0*c/np.sqrt(2.0*self.gamma*(self.gamma-1.0)))* \
                (1.0 - p/state[2])/np.sqrt(1.0 + beta*p/state[2])

        else:
            if p < state[2]:
                # rarefaction
                u = state[1] + s*(2.0*c/(self.gamma-1.0))* \
                    (1.0 - (p/state[2])**((self.gamma-1.0)/(2.0*self.gamma)))
            else:
                # shock
                beta = (self.gamma+1.0)/(self.gamma-1.0)
                u = state[1] + s*(2.0*c/np.sqrt(2.0*self.gamma*(self.gamma-1.0)))* \
                    (1.0 - p/state[2])/np.sqrt(1.0 + beta*p/state[2])

        return u

    def find_star_state(self, p_min=0.001, p_max=1000.0):
        """ root find the Hugoniot curve to find ustar, pstar """

        # we need to root-find on
        self.pstar = brentq(
            lambda p: self.u_hugoniot(p, "left") - self.u_hugoniot(p, "right"),
            p_min, p_max)
        self.ustar = self.u_hugoniot(self.pstar, "left")


    def find_2shock_star_state(self, p_min=0.001, p_max=1000.0):
        """ root find the Hugoniot curve to find ustar, pstar """

        # we need to root-find on
        self.pstar = brentq(
            lambda p: self.u_hugoniot(p, "left", shock=True) - self.u_hugoniot(p, "right", shock=True),
            p_min, p_max)
        self.ustar = self.u_hugoniot(self.pstar, "left", shock=True)

    def shock_solution(self, sgn, xi, state):
        """return the interface solution considering a shock"""

        p_ratio = self.pstar/state[2]
        c = np.sqrt(self.gamma*state[2]/state[0])

        # Toro, eq. 4.52 / 4.59
        S = state[1] + sgn*c*np.sqrt(0.5*(self.gamma + 1.0)/self.gamma*p_ratio +
                                    0.5*(self.gamma - 1.0)/self.gamma)

        # are we to the left or right of the shock?
        if (sgn > 0 and xi > S) or (sgn < 0 and xi < S):
            # R/L region
            solution = state
        else:
            # * region -- get rhostar from Toro, eq. 4.50 / 4.57
            gam_fac = (self.gamma - 1.0)/(self.gamma + 1.0)
            rhostar = state[0] * (p_ratio + gam_fac)/(gam_fac * p_ratio + 1.0)
            solution = np.stack([rhostar, self.ustar, self.pstar], axis=0)

        return solution

    def rarefaction_solution(self, sgn, xi, state):
        """return the interface solution considering a rarefaction wave"""

        # find the speed of the head and tail of the rarefaction fan

        # isentropic (Toro eq. 4.54 / 4.61)
        p_ratio = self.pstar/state[2]
        c = np.sqrt(self.gamma*state[2]/state[0])
        cstar = c*p_ratio**((self.gamma-1.0)/(2*self.gamma))

        lambda_head = state[1] + sgn*c
        lambda_tail = self.ustar + sgn*cstar

        gam_fac = (self.gamma - 1.0)/(self.gamma + 1.0)

        if (sgn > 0 and xi > lambda_head) or (sgn < 0 and xi < lambda_head):
            # R/L region
            solution = state

        elif (sgn > 0 and xi < lambda_tail) or (sgn < 0 and xi > lambda_tail):
            # * region, we use the isentropic density (Toro 4.53 / 4.60)
            solution = np.stack([state[0]*p_ratio**(1.0/self.gamma), self.ustar, self.pstar], axis=0)

        else:
            # we are in the fan -- Toro 4.56 / 4.63
            rho = state[0] * (2/(self.gamma + 1.0) -
                               sgn*gam_fac*(state[1] - xi)/c)**(2.0/(self.gamma-1.0))
            u = 2.0/(self.gamma + 1.0) * ( -sgn*c + 0.5*(self.gamma - 1.0)*state[1] + xi)
            p = state[2] * (2/(self.gamma + 1.0) -
                           sgn*gam_fac*(state[1] - xi)/c)**(2.0*self.gamma/(self.gamma-1.0))
            solution = np.stack([rho, u, p], axis=0)

        return solution

    def sample_solution(self, time, npts, xmin=0.0, xmax=1.0):
        """given the star state (ustar, pstar), sample the solution for npts
        points between xmin and xmax at the given time.

        this is a similarity solution in xi = x/t """

        # we write it all explicitly out here -- this could be vectorized
        # better.

        dx = (xmax - xmin)/npts
        xjump = 0.5*(xmin + xmax)

        x = np.linspace(xmin, xmax, npts, endpoint=False) + 0.5*dx
        xi = (x - xjump)/time

        # which side of the contact are we on?
        # chi = np.sign(xi - self.ustar)

        # gam = self.gamma
        # gam_fac = (gam - 1.0)/(gam + 1.0)

        rho_v = []
        u_v = []
        p_v = []

        for n in range(npts):

            if xi[n] > self.ustar:
                # we are in the R* or R region
                state = self.right
                sgn = 1.0
            else:
                # we are in the L* or L region
                state = self.left
                sgn = -1.0

            # is non-contact wave a shock or rarefaction?
            if self.pstar > state[2]:
                # compression! we are a shock
                solution = self.shock_solution(sgn, xi[n], state)

            else:
                # rarefaction
                solution = self.rarefaction_solution(sgn, xi[n], state)

            # store
            rho_v.append(solution[0])
            u_v.append(solution[1])
            p_v.append(solution[2])

        return x, np.array(rho_v), np.array(u_v), np.array(p_v)

    def _update(self, dt, time):

        x_values = self.grid.real_x
        npts = len(x_values)

        if time == 0:  # xi = (x - xjump)/time  wouldn't work with time == 0
            pass
        else:
            x_e, rho_e, v_e, p_e = self.sample_solution(time, npts)
            u = rho_e * v_e  # convert back to states of the form (rho, rho*u, rho*E)
            e = p_e / (self.gamma - 1) / rho_e
            E = rho_e * (e + 0.5 * v_e ** 2)
            updated_u = np.stack([rho_e, u, E], axis=0)
            self.grid.set(updated_u)

    def _reset(self, params):
        self.grid.reset(params)
        self.left = self.euler_state_conversion(self.grid.space[:, 0])
        self.right = self.euler_state_conversion(self.grid.space[:, -1])

        self.ustar = None
        self.pstar = None
        self.find_star_state()

    def get_full(self):
        return self.grid.get_full()

    def get_real(self):
        return self.grid.get_real()

    def set(self, real_values):
        self.grid.set(real_values)
