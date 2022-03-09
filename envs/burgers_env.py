import gym
import numpy as np
import tensorflow as tf
from gym import spaces

from envs.abstract_pde_env import AbstractPDEEnv
from envs.plottable_env import Plottable1DEnv
from envs.weno_solution import WENOSolution, PreciseWENOSolution, PreciseWENOSolution2D
from envs.weno_solution import lf_flux_split_nd, weno_sub_stencils_nd
from envs.weno_solution import tf_lf_flux_split, tf_weno_sub_stencils, tf_weno_weights
from envs.weno_solution import RKMethod
from envs.solutions import AnalyticalSolution
from envs.solutions import MemoizedSolution, OneStepSolution
import envs.weno_coefficients as weno_coefficients
from envs.interpolate import tf_weno_interpolation
from util.softmax_box import SoftmaxBox
from util.misc import create_stencil_indexes

class AbstractBurgersEnv(AbstractPDEEnv):
    """
    Environment of an N-dimensional Burgers equation.

    This class includes burgers_flux(), which defines the Burgers equation in the space of
    conservation equations.
    This abstract class declares self.solution (and possibly self.weno_solution), the solution for
    the RL agent to compare against. It also adds the viscosity parameter 'nu' by overriding
    _finish_step().

    Concrete subclasses still need to implement render(), _prep_state(), _rk_substep(),
    and declare the observation and action spaces.
    """

    def __init__(self,
            nu=0.0,
            analytical=False, memoize=False,
            precise_weno_order=None, precise_scale=1,
            follow_solution=False,
            solution_rk_method=None,
            *args, **kwargs):
        """
        Construct the Burgers environment.

        Parameters
        ----------
        nu : float
            Viscosity parameter; weight of the 2nd derivative.
        analytical : bool
            Whether to use an analytical solution instead of a numerical one.
        memoize : bool
            Whether to memoize the weno solution.
        precise_weno_order : int
            Use to compute the weno solution with a higher weno order.
        precise_scale : int
            Use to compute the weno solution with more precise grid.
        follow_solution : bool
            Keep the state identical to the solution state, while calculating rewards as if we had
            followed the RL action. Possibly useful for debugging.
        solution_rk_method : RKMethod
            They RK method used by the solution. By default, the same method as used in the
            environment.
        *args, **kwargs
            The remaining arguments are passed to AbstractPDEEnv.
        """
        super().__init__(eqn_type='burgers', *args, **kwargs)

        self.nu = nu

        self.analytical = analytical
        self.memoize = memoize

        if precise_weno_order is None:
            precise_weno_order = self.weno_order
        self.precise_weno_order = precise_weno_order
        self.precise_scale = precise_scale

        self.follow_solution = follow_solution

        if self.analytical:
            if self.grid.ndim > 1:
                raise NotImplementedError("Analytical solutions for multiple dimensions not yet"
                 + " implemented.")
            else:
                self.solution = AnalyticalSolution(self.grid.nx, self.grid.ng,
                        self.grid.xmin, self.grid.xmax, vec_len=1, init_type=self.grid.init_type)
        else:
            if solution_rk_method is None:
                solution_rk_method = self.rk_method
            if self.grid.ndim == 1:
                self.solution = PreciseWENOSolution(
                        self.grid, {'init_type':self.grid.init_type,
                            'boundary':self.grid.boundary},
                        precise_scale=precise_scale, precise_order=precise_weno_order,
                        flux_function=self.burgers_flux, source=self.source,
                        nu=nu, vec_len=1, rk_method=solution_rk_method)
            elif self.grid.ndim == 2:
                self.solution = PreciseWENOSolution2D(
                        self.grid, {'init_type':self.grid.init_type,
                            'boundary':self.grid.boundary},
                        precise_scale=precise_scale, precise_order=precise_weno_order,
                        flux_function=self.burgers_flux, source=self.source,
                        nu=nu, vec_len=1, rk_method=solution_rk_method)
            else:
                raise NotImplementedError("{}-dim solution".format(self.grid.ndim)
                        + " not implemented.")

        if "one-step" in self.reward_mode:
            self.solution = OneStepSolution(self.solution, self.grid, vec_len=1)
            if memoize:
                print("Note: can't memoize solution when using one-step reward.")
        elif memoize:
            self.solution = MemoizedSolution(self.solution, self.episode_length, vec_len=1)

        if self.analytical:
            show_separate_weno = True
            self.solution_label = "True Solution"
            self.weno_solution_label = "WENO"
        elif self.precise_weno_order != self.weno_order or self.precise_scale != 1:
            show_separate_weno = True
            self.solution_label = "WENO (order={}, grid_scale={})".format(self.precise_weno_order,
                    self.precise_scale)
            self.weno_solution_label = "WENO (order={}, grid_scale=1)".format(self.weno_order, self.nx)
        elif isinstance(self.solution, OneStepSolution):
            show_separate_weno = True
            self.solution_label = "WENO (one-step)"
            self.weno_solution_label = "WENO (full)"
        else:
            self.solution_label = "WENO"
            show_separate_weno = False

        if show_separate_weno:
            if self.grid.ndim == 1:
                self.weno_solution = PreciseWENOSolution(
                        self.grid, {'init_type':self.grid.init_type,
                            'boundary':self.grid.boundary},
                        precise_scale=1, precise_order=self.weno_order,
                        flux_function=self.burgers_flux, source=self.source,
                        nu=nu, eqn_type='burgers', vec_len=1, rk_method=self.rk_method)
            elif self.grid.ndim == 2:
                self.weno_solution = PreciseWENOSolution2D(
                        self.grid, {'init_type':self.grid.init_type,
                            'boundary':self.grid.boundary},
                        precise_scale=1, precise_order=self.weno_order,
                        flux_function=self.burgers_flux, source=self.source,
                        nu=nu, eqn_type='burgers', vec_len=1, rk_method=self.rk_method)

            if memoize:
                self.weno_solution = MemoizedSolution(self.weno_solution, self.episode_length, vec_len=1)
        else:
            self.weno_solution = None

    def burgers_flux(self, q):
        return 0.5 * q ** 2

    #def lap(self):
        """
        Returns the Laplacian of grid.u.
        """
        # Note: This function is now defined in envs/grid.py#GridBase.laplacian(). Use that instead.

    def _rk_substep(self, action):
        rhs = super()._rk_substep(action)
        if self.nu > 0.0:
            rhs += self.nu * self.grid.laplacian()
        return rhs

    def _finish_step(self, new_state, dt):
        state, reward, done = super()._finish_step(new_state, dt)

        if self.follow_solution:
            self.grid.set(self.solution.get_real())
            state = self._prep_state()

        return state, reward, done

class WENOBurgersEnv(AbstractBurgersEnv, Plottable1DEnv):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.nx = self.grid.nx

        self.action_space = SoftmaxBox(low=0.0, high=1.0,
                                       shape=(self.grid.real_length() + 1, 2, self.weno_order),
                                       dtype=np.float64)
        self.observation_space = spaces.Box(low=-1e7, high=1e7,
                                            shape=(self.grid.real_length() + 1, 2, 2 * self.state_order - 1),
                                            dtype=np.float64)

        self.solution.set_record_state(True)
        if self.weno_solution is not None:
            self.weno_solution.set_record_state(True)

        if self.weno_solution is not None:
            self.weno_solution.set_record_actions("weno")
        elif not isinstance(self.solution, WENOSolution):
            self.solution.set_record_actions("weno")

        self._action_labels = ["$\omega^{}_{}$".format(sign, num) for sign in ['+', '-']
                                    for num in range(1, self.weno_order+1)]

    def _prep_state(self):
        """
        Return state at current time step. Returns fpr and fml vector slices.

        Returns
        -------
        state: np array.
            State vector to be sent to the policy.
            Size: (grid_size + 1) BY 2 (one for fp and one for fm) BY stencil_size
            Example: order =3, grid = 128

        """
        # compute flux at each point
        u_values = self.grid.get_full()
        flux = self.burgers_flux(u_values)

        fm, fp = lf_flux_split_nd(flux, u_values)

        # Use the first (and only) vector component.
        fm = fm[0]
        fp = fp[0]

        fp_stencil_indexes = create_stencil_indexes(stencil_size=self.state_order * 2 - 1,
                                                    num_stencils=self.grid.real_length() + 1,
                                                    offset=self.grid.ng - self.state_order)
        fm_stencil_indexes = np.flip(fp_stencil_indexes, axis=-1) + 1
        fp_stencils = fp[fp_stencil_indexes]
        fm_stencils = fm[fm_stencil_indexes]

        # Stack fp and fm on axis 1 so grid position is still axis 0.
        state = np.stack([fp_stencils, fm_stencils], axis=1)

        # save this state for convenience
        self.current_state = state

        return state

    def _rk_substep(self, weights):

        state = self.current_state

        fp_state = state[:, 0, :]
        fm_state = state[:, 1, :]

        #TODO state_order != weno_order has never worked well.
        # Is this why? Should this be state order? Or possibly it should be weno order but we still
        # need to compensate for a larger state order somehow?
        fp_stencils = weno_sub_stencils_nd(fp_state, self.weno_order)
        fm_stencils = weno_sub_stencils_nd(fm_state, self.weno_order)

        fp_weights = weights[:, 0, :]
        fm_weights = weights[:, 1, :]

        fpr = np.sum(fp_weights * fp_stencils, axis=-1)
        fml = np.sum(fm_weights * fm_stencils, axis=-1)

        flux = fml + fpr

        rhs = (flux[:-1] - flux[1:]) / self.grid.dx

        # Don't change this to +=; there's a hidden broadcast over the vector dimension.
        rhs = rhs + super()._rk_substep(weights)

        return rhs

    #@tf.function
    def tf_prep_state(self, state):
        # Note the similarity to _prep_state() above.

        # state (the real physical state) does not have ghost cells, but agents operate on a stencil
        # that can spill beyond the boundary, so we need to add ghost cells to create the rl_state.

        boundary = self.grid.boundary
        full_state = self.grid.tf_update_boundary(state, boundary)

        # Compute flux. Burgers specific!!!
        flux = 0.5 * (full_state ** 2)

        flux_minus, flux_plus = tf_lf_flux_split(flux, full_state)

        # Use the first (and only) vector component.
        flux_minus = flux_minus[0]
        flux_plus = flux_plus[0]

        #TODO Could change things to use traditional convolutions instead of going through these
        # stencil indexes to do the same thing.
        plus_indexes = create_stencil_indexes(
                        stencil_size=self.state_order * 2 - 1,
                        num_stencils=self.nx + 1,
                        offset=self.ng - self.state_order)
        minus_indexes = np.flip(plus_indexes, axis=-1) + 1

        plus_stencils = tf.gather(flux_plus, plus_indexes)
        minus_stencils = tf.gather(flux_minus, minus_indexes)

        # Stack together into rl_state.
        # Stack on dim 1 to keep location dim 0.
        rl_state = tf.stack([plus_stencils, minus_stencils], axis=1)

        #return rl_state
        return (rl_state,) # Singleton because this is 1D.

    def tf_rk_substep(self, args):
        real_state, rl_state, rl_action = args
        assert type(rl_state) is tuple and type(rl_action) is tuple

        rl_state = rl_state[0] # Extract 1st (and only) dimension.
        rl_action = rl_action[0]

        # Note that real_state does not contain ghost cells here, but rl_state DOES (and rl_action has
        # weights corresponding to the ghost cells).

        plus_stencils = rl_state[:, 0]
        minus_stencils = rl_state[:, 1]
        plus_action = rl_action[:, 0]
        minus_action = rl_action[:, 1]

        fpr = tf.reduce_sum(plus_action * tf_weno_sub_stencils(plus_stencils, self.weno_order),
                                axis=-1)
        fml = tf.reduce_sum(minus_action * tf_weno_sub_stencils(minus_stencils, self.weno_order),
                                axis=-1)
        reconstructed_flux = fpr + fml

        derivative_u_t = (reconstructed_flux[:-1] - reconstructed_flux[1:]) / self.grid.dx
        rhs = tf.expand_dims(derivative_u_t, axis=0) # Insert the vector dimension.

        if self.nu != 0.0:
            rhs += self.nu * self.grid.tf_laplacian(real_state)
        if self.source != None:
            raise NotImplementedError("External source has not been implemented"
                    + " in global backprop.")

        return rhs

    #@tf.function
    def tf_integrate(self, args):
        real_state, rl_state, rl_action = args
        assert type(rl_state) is tuple and type(rl_action) is tuple

        rhs = self.tf_rk_substep(args)
        dt = self.tf_timestep(real_state)
        step = dt * rhs

        new_state = real_state + step
        return new_state

    # TODO: modify this so less is in subclass?
    # Could put in abstract super class which calls tf_solution() to compute the weno solution,
    # except there are differences related to ghost cells.
    # A future change could be to put the ghost cells in the real state after all.
    # That said, there are still key differences to how different dimensions behave.
    # Perhaps we need separate nD classes?
    #@tf.function
    def tf_calculate_reward(self, args):
        real_state, rl_state, rl_action, next_real_state = args
        assert type(rl_state) is tuple and type(rl_action) is tuple
        # Note that real_state and next_real_state do not include ghost cells, but rl_state does.


        if "conserve" in reward_mode:
            if "Burgers" not in str(self) or self.dimensions > 1:
                raise Exception("conservation reward not implemented for this environment")
            match = re.search("n(\d+)", self.reward_mode)
            if match is None:
                points = 1
            else:
                points = match.group(1)

            # Remove vector dimension.
            real_state = real_state[0]
            next_real_state = next_real_state[0]

            # Compute boundaries.
            boundary = self.grid.boundary
            real_state_full = self.grid.tf_update_boundary(real_state, boundary)
            next_real_state_full = self.grid.tf_update_boundary(next_real_state, boundary)

            # Interpolate.
            # TODO: calculate WENO weights instead of using the default Burgers ones.
            real_state_interpolated = tf_weno_interpolation(real_state, weno_order=self.weno_order,
                    points=points, num_ghosts=self.grid.ng)
            next_real_state_interpolated = tf_weno_interpolation(real_state,
                    weno_order=self.weno_order, points=points, num_ghosts=self.grid.ng)

            previous_total = tf.reduce_sum(real_state_interpolated)
            current_total = tf.reduce_sum(next_real_state_interpolated)

            reward = tf.atan(-tf.abs(current_total - previous_total))
            return reward, done

        rl_state = rl_state[0] # Extract 1st (and only) dimension.
        rl_action = rl_action[0]

        rk_method = self.solution.rk_method
        if rk_method == RKMethod.EULER:
            fp_stencils = rl_state[:, 0]
            fm_stencils = rl_state[:, 1]
            fp_weights = tf_weno_weights(fp_stencils, self.weno_order)
            fm_weights = tf_weno_weights(fm_stencils, self.weno_order)
            weno_action = tf.stack([fp_weights, fm_weights], axis=1)
            weno_next_real_state = self.tf_integrate((real_state, (rl_state,), (weno_action,)))
        elif rk_method == RKMethod.RK4:
            next_rl_state = rl_state
            weno_substeps = []
            dt = self.tf_timestep(real_state)
            for stage in range(4):
                fp_stencils = next_rl_state[:, 0]
                fm_stencils = next_rl_state[:, 1]
                fp_weights = tf_weno_weights(fp_stencils, self.weno_order)
                fm_weights = tf_weno_weights(fm_stencils, self.weno_order)
                weno_action = tf.stack([fp_weights, fm_weights], axis=1)
                rhs = self.tf_rk_substep((real_state, (rl_state,), (weno_action,)))
                step = dt * rhs
                weno_substeps.append(step)

                if stage == 0:
                    weno_next_real_state = real_state + step/2
                elif stage == 1:
                    weno_next_real_state = real_state + step/2
                elif stage == 2:
                    weno_next_real_state = real_state + step
                elif stage == 3:
                    weno_next_real_state = real_state + (weno_substeps[0]
                                                        + 2*weno_substeps[1]
                                                        + 2*weno_substeps[2]
                                                        + weno_substeps[3]) / 6

                if stage < 3:
                    next_rl_state = self.tf_prep_state(weno_next_real_state)
                    next_rl_state = next_rl_state[0]
        else:
            raise Exception(f"{rk_method} not implemented.")

        # This section is adapted from AbstactPDEEnv.calculate_reward()
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
        error = tf.reduce_sum(error, axis=0)

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

        boundary = self.grid.boundary
        if not type(boundary) is str:
            boundary = boundary[0]

        # Average of error in two adjacent cells.
        if "adjacent" in self.reward_mode and "avg" in self.reward_mode:
            error = tf.abs(error)
            combined_error = (error[:-1] + error[1:]) / 2
            if boundary == "outflow":
                # Error beyond boundaries will be identical to error at edge, so average error
                # at edge interfaces is just the error at the edge.
                combined_error = tf.concat([[error[0]], combined_error, [error[-1]]], axis=0)
            elif boundary == "periodic":
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
            if boundary == "outflow":
                left_error = tf.fill(ghost_size, error[0])
                right_error = tf.fill(ghost_size, error[-1])
            elif boundary == "periodic":
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

        # We don't have this block from calculate_reward():
        #if np.max(error) > 1e7 or np.isnan(np.max(error)):
            #reward = tuple(reward_part - max_penalty * (self.episode_length - self.steps)
                        #for reward_part in reward)
            #done = True
        # This stops the episode when error is beyond a threshold. How might we implement this in
        # global backprop? That requires episodes of the same length every time.
        # I guess we could make all states and rewards 0s after hitting the error threshold?

        #return reward
        return (reward,) # Singleton because this is 1D.

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

    def step(self, action):
        plus_actions = action // len(self.action_list)
        minus_actions = action % len(self.action_list)

        combined_actions = np.vstack((plus_actions, minus_actions)).transpose()

        selected_actions = self.action_list[combined_actions]

        return super().step(selected_actions)

class SplitFluxBurgersEnv(AbstractBurgersEnv, Plottable1DEnv):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        raise Exception("SplitFluxBurgersEnv hasn't been used in a while! This exception is safe"
                + " to delete, but keep an eye out for unexpected behavior.")

        self.action_space = SoftmaxBox(low=-np.inf, high=np.inf,
                                       shape=(self.grid.real_length() + 1, 2, 2 * self.weno_order - 1),
                                       dtype=np.float64)
        self.observation_space = spaces.Box(low=-1e7, high=1e7,
                                            shape=(self.grid.real_length() + 1, 2, 2 * self.state_order - 1),
                                            dtype=np.float64)
        if self.weno_solution is None:
            self.solution.set_record_actions("coef")
        else:
            self.weno_solution.set_record_actions("coef")

    def _prep_state(self):
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
        f = self.burgers_flux(g.u[0])

        # get maximum velocity
        alpha = np.max(abs(g.u[0]))

        # Lax Friedrichs Flux Splitting
        fp = (f + alpha * g.u[0]) / 2
        fm = (f - alpha * g.u[0]) / 2

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

    def _rk_substep(self, action):
        state = self.current_state

        fp_state = state[:, 0, :]
        fm_state = state[:, 1, :]

        fp_action = action[:, 0, :]
        fm_action = action[:, 1, :]

        fpr = np.sum(fp_action * fp_state, axis=-1)
        fml = np.sum(fm_action * fm_state, axis=-1)

        flux = fml + fpr

        rhs = (flux[:-1] - flux[1:]) / self.grid.dx

        return rhs

class FluxBurgersEnv(AbstractBurgersEnv):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        raise Exception("FluxBurgersEnv hasn't been used in a while! This exception is safe"
                + " to delete, but keep an eye out for unexpected behavior.")

        self.action_space = SoftmaxBox(low=-np.inf, high=np.inf,
                                       shape=(self.grid.real_length() + 1, 2 * self.weno_order),
                                       dtype=np.float64)
        self.observation_space = spaces.Box(low=-1e7, high=1e7,
                                            shape=(self.grid.real_length() + 1, 2 * self.state_order),
                                            dtype=np.float64)

    def _prep_state(self):
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
        f = self.burgers_flux(g.u[0])

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

    def _rk_step(self, action):
        state = self.current_state
        flux = np.sum(action * state, axis=-1)
        rhs = (flux[:-1] - flux[1:]) / self.grid.dx

        return rhs
