import gym
import numpy as np
import tensorflow as tf

from envs.grid import create_grid
from envs.solutions import OneStepSolution
from envs.weno_solution import WENOSolution
from envs.source import RandomSource
from util.misc import create_stencil_indexes
from util.misc import AxisSlice

#TODO adapt to vector quantities. What would need to change?
# Will state be a tuple like the action? Or is it better to have the vector as the last dimension?
class AbstractScalarEnv(gym.Env):
    """
    Environment modelling a scalar conservation equation of arbitrary dimensions.

    This abstract class handles the underlying spatial grid, (in self.grid),
    functions related to indexing the state and action history, calculating the reward signal, 
    the general structure of stepping through time, and resetting the environment.

    The remaining behavior should be handled by one or more levels of subclass.
    In particular a concrete subclass should:
     - Declare self.solution, an instance of SolutionBase, from which the reward is calculated.
     - Optionally declare self.weno_solution, an instance of SolutionBase that closely follows WENO
       for comparison.
       - Also self.solution_label and self.weno_solution_label, as descriptions of what they
         actually are.
     - Declare self.observation_space and self.action_space.
     - Implement render(), which displays the state in some way.
     - Implement _prep_state(), which calculates and returns the RL state representation of the
       grid.
     - Implement _rk_substep(), which calculates the next step based on the action.
     - Optionally override step() and _finish_step(), if something else needs to happen before or
       after the step respectively.
     - Optionally override reset(), if something else needs to happen on each new episode.
    """

    def __init__(self,
            num_cells=(128, 128), num_ghosts=None, min_value=0.0, max_value=1.0,
            boundary=None, init_type="gaussian",
            init_params=None,
            fixed_step=0.0004, C=None, #C=0.5,
            weno_order=3, state_order=None, srca=0.0, episode_length=250,
            reward_adjustment=1000, reward_mode=None,
            test=False):
        """
        Construct the environment.

        Parameters
        ----------
        num_cells : int OR [int]
            Number of discretized cells.
        num_ghosts: int OR [int]
            Number of ghosts. Default is weno_order + 1, which is the min required for WENO to
            work.
        min_value : float OR [float]
            Lower bounds of the physical space.
        max_value : float OR [float]
            Upper bounds of the physical space.
        boundary : string OR [string]
            Type of boundary condition (periodic/outflow).
        init_type : string
            Type of initial condition (various, see envs.grid).
        init_params : dict
            Parameters dict to send to grid.reset, overriding initial defaults.
            Useful for specifying minor parameters like equation constants.
        fixed_step : float
            Length of fixed timestep. Fixed timesteps are used by default.
        C : float
            CFL number for varying length timesteps. None by default. Specify to use varying length
            timesteps.
        weno_order : int
            Order of WENO approximation. Affects action and state space sizes.
        state_order : int
            Use to specify a state space that is wider than it otherwise should be.
        srca : float
            Amplitude of random external source.
        episode_length : int
            Number of timesteps before the episode terminates.
        reward_adjustment : float
            Adjust the reward squashing function - higher makes small changes to large rewards
            less important.
        reward_mode : str
            Composite string specifying the type of reward function.
        test : bool
            Whether this is a strictly test environment. This is passed to grid, which can ensure initial
            conditions are not randomized.
        """

        self.test = test
        
        self.reward_mode = self.fill_default_reward_mode(reward_mode)
        if reward_mode != self.reward_mode:
            print("Reward mode updated to '{}'.".format(self.reward_mode))

        #self.nx = nx
        self.weno_order = weno_order
        if state_order is None: # Does state_order go in this class?
            self.state_order = self.weno_order
        else:
            self.state_order = self.weno_order

        if num_ghosts is None:
            self.ng = self.state_order + 1
        else:
            self.ng = num_ghosts

        if type(num_cells) is int:
            dims = 1
        else:
            dims = len(num_cells)

        self.grid = create_grid(dims,
                                num_cells=num_cells, min_value=min_value, max_value=max_value,
                                num_ghosts=self.ng,
                                boundary=boundary, init_type=init_type,
                                deterministic_init=self.test)
        self.init_params = init_params

        if srca > 0.0:
            if self.grid.ndim > 1:
                raise NotImplementedError("Random source not implemented for multiple dimensions.")
            self.source = RandomSource(grid=self.grid, amplitude=srca)
        else:
            self.source = None

        # Subclass must declare a solution. It should be an instance of SolutionBase.
        self.solution = None
        self.solution_label = "solution"
        # Subclass may optionally use a weno_solution. This is for keeping track of the comparison
        # solution when the main solution is doing something else. This could use a more general
        # name.
        self.weno_solution = None
        self.weno_solution_label = "WENO"

        self.previous_error = np.zeros_like(self.grid.get_full())

        self.fixed_step = fixed_step
        self.C = C  # CFL number # Why this comment? Why not just call it cfl or CFL?
        self.episode_length = episode_length
        self.reward_adjustment = reward_adjustment

        # Useful values for subclasses to create consistent names.
        self._step_precision = int(np.ceil(np.log(1+self.episode_length) / np.log(10)))
        self._cell_index_precision = int(np.ceil(np.log(1+max(self.grid.num_cells)) / np.log(10)))

        self.t = 0.0
        self.steps = 0
        self.action_history = []
        self.state_history = []

        # These are used by RK4.
        self.dt = self.timestep()
        self.k1 = self.k2 = self.k3 = self.u_start = None
        self.rk_state = 1

        # Subclass should declare the observation and action spaces.
        self.observation_space = None
        self.action_space = None

    def render(self, *args, **kwargs):
        """
        Display some representation of the current state.
        """
        raise NotImplementedError()

    def _prep_state(self):
        """
        Calculate and return the RL state from self.grid.

        Should also set self.state with the RL state.
        """
        raise NotImplementedError()

    def _rk_substep(self, action):
        """
        Calculate and return the direction of change of the space based on the action.

        The actual change is dt * self._rk_substep(action); this function should not calculate the
        timestep.
        The return value may be applied on its own for Euler stepping, or used as part of a
        Runge-Kutta method.
        """
        raise NotImplementedError()

    @property
    def dimensions(self): return self.grid.ndim

    def step(self, action):
        """
        Perform a single time step.

        Parameters
        ----------
        action : ndarray OR [ndarray] OR ?
          Weights for advancing the simulation.
          The shape depends on the action space in the subclass.

        Returns
        -------
        state : ndarray OR [ndarray] OR ?
          RL state representation of the solution computed from the action.
          The shape depends on the observation space in the subclass.
        reward: ndarray OR [ndarray]
          reward for current action
        done : bool
          boolean signifying end of episode
        info : dict
          Additional information; nothing currently.
        """
        assert not (np.isnan(action).any() if self.grid.ndim == 1 else \
                any([np.isnan(a).any() for a in action])), "NaN detected in action."

        self.action_history.append(action)

        dt = self.timestep()

        step = dt * self._rk_substep(action)

        state, reward, done = self._finish_step(step, dt)

        self.state_history.append(self.grid.get_full().copy())

        return state, reward, done, {}

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
        assert not (np.isnan(action).any() if self.grid.ndim == 1 else \
                any([np.isnan(a).any() for a in action])), "NaN detected in action."

        if self.rk_state == 1:
            self.u_start = np.array(self.grid.get_real())
            self.dt = self.timestep()

            self.k1 = self.dt * self.rk_substep_weno(action)
            self.grid.set(self.u_start + self.k1/2)
            state = self._prep_state()

            self.rk_state = 2
            return state, np.zeros_like(action), False
        elif self.rk_state == 2:
            self.k2 = self.dt * self.rk_substep_weno(action)
            self.grid.set(self.u_start + self.k2/2)
            state = self._prep_state()

            self.rk_state = 3
            return state, np.zeros_like(action), False
        elif self.rk_state == 3:
            self.k3 = self.dt * self.rk_substep_weno(action)
            self.grid.set(self.u_start + self.k3)
            state = self._prep_state()

            self.rk_state = 4
            return state, np.zeros_like(action), False
        else:
            assert self.rk_state == 4

            self.action_history.append(action)

            k4 = self.dt * self.rk_substep_weno(action)
            step = (self.k1 + 2*(self.k2 + self.k3) + k4) / 6

            if isinstance(self.solution, WENOSolution):
                self.solution.set_rk4(True)
            if self.weno_solution is not None:
                self.weno_solution.set_rk4(True)

            state, reward, done = self._finish_step(step, self.dt, prev=self.u_start)

            if isinstance(self.solution, WENOSolution):
                self.solution.set_rk4(False)
            if self.weno_solution is not None:
                self.weno_solution.set_rk4(False)

            self.state_history.append(self.grid.get_full().copy())

            self.rk_state = 1
            self.k1 = self.k2 = self.k3 = self.u_start = self.dt = None
            return state, reward, done

    def _finish_step(self, step, dt, prev=None):
        """
        Apply a physical step.

        If prev is None, the step is applied to the current grid, otherwise
        it is applied to prev, then saved to the grid.

        Override this function if other adjustments should be done to the step before applying it
        to the grid. (Probably call super()._finish_step() at the END of that method.)

        Returns state, reward, done.
        """
        # Update solution before updating the grid, in case it depends on the current grid.
        self.solution.update(dt, self.t)
        if self.weno_solution is not None:
            self.weno_solution.update(dt, self.t)

        if self.source is not None:
            self.source.update(dt, self.t + dt) # Why is this t + dt? Why is source based on the
                                                # time after we integrate forward?
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

        state = self._prep_state()

        assert not (np.isnan(state).any() if self.grid.ndim == 1 else \
                any([np.isnan(s).any() for s in state])), "NaN detected in state."
        assert not (np.isnan(reward).any() if self.grid.ndim == 1 else \
                any([np.isnan(r).any() for r in reward])), "NaN detected in reward."

        return state, reward, done

    # Works in ND.
    def reset(self):
        """
        Reset the environment.

        This is the abstract version. In a concrete subclass, the overriding function should call this
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

        self.rk_state = 1

        return self._prep_state()

    # Works in ND.
    @staticmethod
    def fill_default_time_vs_space(num_cells, min_value, max_value, dt, C, ep_length, time_max):
        approximate_max = 2.0 # Is this a good number?
        if num_cells is None:
            if dt is None:
                dt = 0.0004
            cell_size = dt * approximate_max / C
            num_cells = tuple(int(np.ceil((xmax - xmin) / cell_size))
                                    for xmin, xmax in zip(min_value, max_value))
        elif dt is None:
            min_cell_size = min((xmax - xmin) / nx for xmin, xmax, nx
                                                    in zip(min_value, max_value, num_cells))
            dt = C * min_cell_size / approximate_max

        if ep_length is None:
            if time_max is None:
                ep_length = 500
            else:
                ep_length = int(np.ceil(time_max / dt))

        return num_cells, dt, ep_length

    # Works in ND.
    def timestep(self):
        if self.C is None:  # return a constant time step
            return self.fixed_step
        else:
            min_cell_size = min(self.grid.cell_size)
            return self.C * min_cell_size / max(abs(self.grid.get_real()))


    # Works in ND.
    def force_state(self, state_grid):
        """
        Override the current state with something else.

        You should have a unusual reason for doing this, like if you need to copy the state from a
        different environment.
        state and action history will not make sense after calling this.

        Parameters
        ----------
        state_grid : ndarray
            Array of new state values. Should not include ghost cells.
        """
        self.grid.set(state_grid)
        # Rewrite recent history.
        self.state_history[-1] = self.grid.get_full().copy()

    # Works in ND.
    def get_state(self, timestep=None, location=None, full=True):
        assert timestep is None or location is None

        if timestep is None and location is None:
            state = self.grid.get_full() if full else self.grid.get_real()
        elif timestep is not None:
            state = np.array(self.state_history)[timestep]
            if not full:
                state = state[self.grid.real_slice]
        else:
            if type(location) is int:
                location = (location,)
            # This is equivalent to [:, location] except location is a tuple.
            state = np.array(self.state_history)[(slice(None),) + location]
        return state

    # Works in ND.
    def get_solution_state(self, timestep=None, location=None, full=True):
        assert timestep is None or location is None

        #TODO: Does it make sense to return the self.weno_solution state instead if using a
        # one-step reward? (That is, a OneStepSolution?)

        if timestep is None and location is None:
            state = self.solution.get_full() if full else self.solution.get_real()
        elif timestep is not None:
            state = np.array(self.solution.get_state_history())[timestep]
            if not full:
                state = state[self.grid.real_slice]
        else:
            if type(location) is int:
                location = (location,)
            state = np.array(self.solution.get_state_history())[slice(None,) + location]
        return state

    # Works in ND.
    def get_error(self, timestep=None, location=None, full=True):
        return (self.get_state(timestep, location, full) 
                - self.get_solution_state(timestep, location, full))

    # Works in ND.
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
            combined_cell_size = np.prod(self.grid.cell_size)
            l2_error = np.sqrt(combined_cell_size * np.sum(np.square(error)))
            return l2_error

    # Works in ND.
    def get_action(self, timestep=None, location=None, axis=None,
            action_history=None):
        """
        Return the actions for a given time or space. By default return the most recent action.

        This function cannot index a particular time and space; nor will it index all of time and
        space (if that is needed use env.action_history directly).

        In other words, either timestep or location must be left as the default None. 
        If both are None, timestep instead defaults to the most recent timestep.

        Parameters
        ----------
        timestep : int
            Select the actions at the timestep. Defaults to the most recent timestep.
        location : int or tuple of int
            Select the actions at the location.
        axis : int
            For multi-dimensional environments, select only the actions along the given axis.
            This argument is ignored for one-dimensional environments.
        action_history : list
            Override the internal action_history with a different action_history.
            Possibly useful if you've copied the action_history from an earlier run.

        Returns
        -------
        actions : ?
            The actions at the selected time/location. The shape depends on the action shape of
            this environment.
        """
        assert timestep is None or location is None

        if action_history is None:
            action_history = self.action_history

        if timestep is None and location is None:
            action = action_history[-1]
            if axis is not None:
                action = action[axis]
        elif timestep is not None:
            action = action_history[timestep]
            if axis is not None:
                action = action[axis]
        else:
            if type(action_history[0]) is tuple:
                if axis is None:
                    action = [tuple(a_x[slice(None,) + location] for
                                a_x in a) for a in action_history]
                else:
                    action = [a[axis][slice(None,) + location] for a in action_history]
            else:
                action = action_history[:, location]
        return action

    # Works in ND.
    def get_solution_action(self, timestep=None, location=None, axis=None):
        """
        Returns the action of the solution. Arguments act the same as get_action().
        """
        assert self.solution.is_recording_actions()

        solution_action_history = self.solution.get_action_history()
        return self.get_action(timestep, location, axis, action_history=solution_action_history)

    # Works in ND.
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
    
    # Works in ND.
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
            #TODO This should probably trim ghosts from other axes.
            combined_error = tuple((AxisSlice(error, axis)[ng-1:ng]
                                + AxisSlice(error, axis)[ng:-(ng-1)]) / 2
                                    for axis, ng in enumerate(self.grid.num_ghosts))
        # Combine error across the WENO stencil.
        # (NOT the state stencil i.e. self.state_order * 2 - 1, even if we are using a wide state.)
        elif "stencil" in self.reward_mode:
            combined_error = []
            for axis, (nx, ng) in enumerate(zip(self.grid.num_cells, self.grid.num_ghosts)):
                stencil_indexes = create_stencil_indexes(
                        stencil_size=(self.weno_order * 2 - 1),
                        num_stencils=(nx + 1),
                        offset=(ng - self.weno_order))
                error_stencils = AxisSlice(error, axis)[stencil_indexes]
                error_stencils = error[stencil_indexes]
                if "max" in self.reward_mode:
                    combined_error.append(np.amax(error_stencils, axis=-1))
                elif "avg" in self.reward_mode:
                    combined_error.append(np.mean(error_stencils, axis=-1))
                elif "L2" in self.reward_mode:
                    combined_error.append(np.sqrt(np.sum(error_stencils**2, axis=-1)))
                else:
                    raise Exception("AbstractBurgersEnv: reward_mode problem")
        else:
            raise Exception("AbstractBurgersEnv: reward_mode problem")

        # Squash reward.
        if "nosquash" in self.reward_mode:
            max_penalty = 1e7
            reward = tuple(error for error in combined_error)
        elif "logsquash" in self.reward_mode:
            max_penalty = 1e7
            reward = tuple(np.log(error + 1e-30) for error in combined_error)
        elif "arctansquash" in self.reward_mode:
            max_penalty = np.pi / 2
            if "noadjust" in self.reward_mode:
                reward = tuple(np.arctan(-error) for error in combined_error)
            else:
                # The constant controls the relative importance of small rewards compared to large rewards.
                # Towards infinity, all rewards (or penalties) are equally important.
                # Towards 0, small rewards are increasingly less important.
                # An alternative to arctan(C*x) with this property would be x^(1/C).
                reward = tuple(np.arctan(self.reward_adjustment * -error) for error in
                        combined_error)
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

        if self.grid.ndim == 1:
            reward = reward[0]

        return reward, done

    #TODO The documentation on these may need to change for ND.
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
 
    # Works with ND.
    def close(self):
        # Delete references for easier garbage collection.
        self.grid = None
        self.solution = None
        self.weno_solution = None
        self.state_history = []
        self.action_history = []

    # Works with ND.
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
        # The official Env class has this as part of its interface, but I don't think we need it.
        # Better to set the numpy seed at the experiment level.
        pass
