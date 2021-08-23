import numpy as np
import tensorflow as tf
from gym import spaces

import envs.weno_coefficients as weno_coefficients
from envs.abstract_pde_env import AbstractPDEEnv
from envs.plottable_env import Plottable1DEnv
from envs.solutions import AnalyticalSolution
from envs.solutions import MemoizedSolution, OneStepSolution
from envs.weno_solution import WENOSolution, PreciseWENOSolution, PreciseWENOSolution2D
from envs.weno_solution import lf_flux_split_nd, weno_sub_stencils_nd
from envs.weno_solution import tf_lf_flux_split, tf_weno_sub_stencils, tf_weno_weights
from util.misc import create_stencil_indexes
from util.softmax_box import SoftmaxBox


class AbstractEulerEnv(AbstractPDEEnv):
    """
    Environment of an N-dimensional Euler equation.

    This class includes euler_flux(), which defines the Euler equation in the space of
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
            *args, **kwargs):
        """
        Construct the Euler environment.

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
        *args, **kwargs
            The remaining arguments are passed to AbstractPDEEnv.
        """
        super().__init__(eqn_type='euler', *args, **kwargs)

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
                        self.grid.xmin, self.grid.xmax, vec_len=3, init_type=init_type)
        else:
            if self.grid.ndim == 1:
                self.solution = PreciseWENOSolution(
                        self.grid, {'init_type':self.grid.init_type,
                            'boundary':self.grid.boundary},
                        precise_scale=precise_scale, precise_order=precise_weno_order,
                        flux_function=self.euler_flux, source=self.source,
                        nu=nu, eqn_type='euler', vec_len=3)
            # elif self.grid.ndim == 2:
            #     self.solution = PreciseWENOSolution2D(
            #             self.grid, {'init_type':self.grid.init_type,
            #                 'boundary':self.grid.boundary},
            #             precise_scale=precise_scale, precise_order=precise_weno_order,
            #             flux_function=self.burgers_flux, source=self.source,
            #             nu=nu, eqn_type='euler', vec_len=3)
            else:
                raise NotImplementedError("{}-dim solution".format(self.grid.ndim)
                        + " not implemented.")

        if "one-step" in self.reward_mode:
            self.solution = OneStepSolution(self.solution, self.grid, vec_len=3)
            if memoize:
                print("Note: can't memoize solution when using one-step reward.")
        elif memoize:
            self.solution = MemoizedSolution(self.solution, self.episode_length, vec_len=3)

        if self.analytical:
            show_separate_weno = True
            self.solution_label = "Analytical"
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
            show_separate_weno = False

        if show_separate_weno:
            if self.grid.ndim == 1:
                self.weno_solution = PreciseWENOSolution(
                        self.grid, {'init_type':self.grid.init_type,
                            'boundary':self.grid.boundary},
                        precise_scale=1, precise_order=self.weno_order,
                        flux_function=self.euler_flux, source=self.source,
                        nu=nu,  eqn_type='euler', vec_len=3)
            # elif self.grid.ndim == 2:
            #     self.weno_solution = PreciseWENOSolution2D(
            #             self.grid, {'init_type':self.grid.init_type,
            #                 'boundary':self.grid.boundary},
            #             precise_scale=1, precise_order=self.weno_order,
            #             flux_function=self.euler_flux, source=self.source,
            #             nu=nu,  eqn_type='euler', vec_len=3)

            if memoize:
                self.weno_solution = MemoizedSolution(self.weno_solution, self.episode_length, vec_len=3)
        else:
            self.weno_solution = None

    def euler_flux(self, q):
        flux = np.zeros_like(q)
        rho = q[0, :]
        S = q[1, :]
        E = q[2, :]
        v = S / rho
        p = (self.eos_gamma - 1) * (E - rho * v ** 2 / 2)
        flux[0, :] = S
        flux[1, :] = S * v + p
        flux[2, :] = (E + p) * v
        return flux

    def _finish_step(self, step, dt, prev=None):
        if self.nu > 0.0:
            R = self.nu * self.grid.laplacian()
            step += dt * R

        state, reward, done = super()._finish_step(step, dt, prev)

        if self.follow_solution:
            self.grid.set(self.solution.get_real())
            state = self._prep_state()

        return state, reward, done



class WENOEulerEnv(AbstractEulerEnv, Plottable1DEnv):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.nx = self.grid.nx

        # For Euler Equation, 0-th dimension for action and observation are hardcoded to be 3
        self.action_space = SoftmaxBox(low=0.0, high=1.0,
                                       shape=(3, self.grid.real_length() + 1, 2, self.weno_order),
                                       dtype=np.float64)
        self.observation_space = spaces.Box(low=-1e7, high=1e7,
                                            shape=(3, self.grid.real_length() + 1, 2, 2 * self.state_order - 1),
                                            dtype=np.float64)

        self.solution.set_record_state(True)
        if self.weno_solution is not None:
            self.weno_solution.set_record_state(True)

        if self.weno_solution is not None:
            self.weno_solution.set_record_actions("weno")
        elif not isinstance(self.solution, WENOSolution):
            self.solution.set_record_actions("weno")

        self._action_labels = ["$w^{}_{}$".format(sign, num) for sign in ['+', '-']
                               for num in range(1, self.weno_order + 1)]

        if 'eos_gamma' in self.init_params:
            self.eos_gamma = self.init_params['eos_gamma']
        else:
            self.eos_gamma = 1.4  # Gamma law EOS
        if 'reconstruction' in self.init_params:
            self.reconstruction = 'characteristicwise' if self.init_params['reconstruction'] else 'componentwise'
            # 0 for 'componentwise', 1 for 'characterwise'
        else:
            self.reconstruction = 'componentwise'

    def _prep_state(self):
        """
        Return state at current time step. Returns fpr and fml vector slices.

        Returns
        -------
        state: np array.
            State vector to be sent to the policy.
            Size: (grid_size + 1) BY 2 (one for fp and one for fm) BY stencil_size
            Example: order = 3, grid = 128

        """
        # get the solution data
        g = self.grid

        self.current_state = []

        # compute flux at each point
        f = self.euler_flux(g.u)
        # get maximum velocity
        alpha = self._max_lambda()

        for i in range(g.space.shape[0]):
            # Lax Friedrichs Flux Splitting
            fp = (f[i] + alpha * g.u[i]) / 2
            fm = (f[i] - alpha * g.u[i]) / 2

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
            self.current_state.append(state)

        self.current_state = np.array(self.current_state)

        return self.current_state

    def _max_lambda(self):
        rho = self.grid.u[0]
        v = self.grid.u[1] / rho
        p = (self.eos_gamma - 1) * (self.grid.u[2, :] - rho * v ** 2 / 2)
        cs = np.sqrt(self.eos_gamma * p / rho)
        return max(np.abs(v) + cs)

    def _rk_substep(self, weights):

        state = self.current_state

        rhs = []
        for i in range(self.grid.space.shape[0]):  # TODO: check if we can get rid of for -yiwei
            fp_state = state[i, :, 0, :]
            fm_state = state[i, :, 1, :]

            # TODO state_order != weno_order has never worked well.
            # Is this why? Should this be state order? Or possibly it should be weno order but we still
            # need to compensate for a larger state order somehow?
            fp_stencils = weno_sub_stencils_nd(fp_state, self.weno_order)
            fm_stencils = weno_sub_stencils_nd(fm_state, self.weno_order)

            fp_weights = weights[i, :, 0, :]
            fm_weights = weights[i, :, 1, :]

            fpr = np.sum(fp_weights * fp_stencils, axis=-1)
            fml = np.sum(fm_weights * fm_stencils, axis=-1)

            flux = fml + fpr

            rhs.append((flux[:-1] - flux[1:]) / self.grid.dx)

        rhs = np.array(rhs)

        return rhs

    # @tf.function
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
            left_ghost = tf.repeat(tf.expand_dims(state[:, 0], 1), self.ng, axis=1)
            right_ghost = tf.repeat(tf.expand_dims(state[:, -1], 1), self.ng, axis=1)
            full_state = tf.concat([left_ghost, state, right_ghost], axis=1)
        elif self.grid.boundary == "periodic":
            left_ghost = state[:, -ghost_size[0]:]
            right_ghost = state[:, :ghost_size[0]]
            full_state = tf.concat([left_ghost, state, right_ghost], axis=1)
        else:
            raise NotImplementedError("{} boundary not implemented.".format(self.grid.boundary))

        # Compute flux. Euler specific!!!
        rho = full_state[0, :]
        S = full_state[1, :]
        E = full_state[2, :]
        v = S / rho
        p = (self.eos_gamma - 1) * (E - rho * v ** 2 / 2)
        flux = tf.stack([S, S * v + p, (E + p) * v], axis=0)

        cs = tf.sqrt(self.eos_gamma * p / rho)
        alpha = tf.reduce_max(tf.abs(v) + cs)

        plus_stencils = []
        minus_stencils = []
        for i in range(self.grid.space.shape[0]):
            flux_plus = (flux[i] + alpha * full_state[i]) / 2
            flux_minus = (flux[i] - alpha * full_state[i]) / 2

            # TODO Could change things to use traditional convolutions instead.
            # Maybe if this whole thing actually works.

            plus_indexes = create_stencil_indexes(
                stencil_size=self.state_order * 2 - 1,
                num_stencils=self.nx + 1,
                offset=self.ng - self.state_order)
            # plus_indexes = create_stencil_indexes(
            # stencil_size=(self.weno_order * 2 - 1),
            # num_stencils=(self.nx + 1),
            # offset=(self.ng - self.weno_order))
            minus_indexes = plus_indexes + 1
            minus_indexes = np.flip(minus_indexes, axis=-1)

            plus_stencils.append(tf.gather(flux_plus, plus_indexes))
            minus_stencils.append(tf.gather(flux_minus, minus_indexes))

        # Stack together into rl_state.
        # Stack on dim 2 to keep location dim 1, vec_len dim 0.
        rl_state = tf.stack([plus_stencils, minus_stencils], axis=2)

        #return rl_state
        return (rl_state,) # Singleton because this is 1D.

    # @tf.function
    def tf_integrate(self, args):
        real_state, rl_state, rl_action = args

        rl_state = rl_state[0] # Extract 1st (and only) dimension.
        rl_action = rl_action[0]

        # Note that real_state does not contain ghost cells here, but rl_state DOES (and rl_action has
        # weights corresponding to the ghost cells).
        new_state = []
        for i in range(self.grid.space.shape[0]):
            plus_stencils = rl_state[i, :, 0, :]
            minus_stencils = rl_state[i, :, 1, :]

            # This block corresponds to envs/weno_solution.py#weno_sub_stencils_nd().
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

            plus_action = rl_action[i, :, 0]
            minus_action = rl_action[i, :, 1]

            fpr = tf.reduce_sum(plus_action * plus_interpolated, axis=-1)
            fml = tf.reduce_sum(minus_action * minus_interpolated, axis=-1)

            reconstructed_flux = fpr + fml

            derivative_u_t = (reconstructed_flux[:-1] - reconstructed_flux[1:]) / self.grid.dx

            # TODO implement RK4?

            step = self.dt * derivative_u_t

            # if self.nu != 0.0:
            #     # Compute the Laplacian. This involves the first ghost cell past the boundary.
            #     central_lap = (real_state[:-2] - 2.0 * real_state[1:-1] + real_state[2:]) / (self.grid.dx ** 2)
            #     if self.grid.boundary == "outflow":
            #         left_lap = (-real_state[0] + real_state[1]) / (self.grid.dx ** 2)  # X-2X+Y = -X+Y
            #         right_lap = (real_state[-2] - real_state[-1]) / (self.grid.dx ** 2)  # X-2Y+Y = X-Y
            #     elif self.grid.boundary == "periodic":
            #         left_lap = (real_state[-1] - 2.0 * real_state[0] + real_state[1]) / (self.grid.dx ** 2)
            #         right_lap = (real_state[-2] - 2.0 * real_state[-1] + real_state[0]) / (self.grid.dx ** 2)
            #     else:
            #         raise NotImplementedError()
            #     lap = tf.concat([[left_lap], central_lap, [right_lap]], axis=0)
            #
            #     step += self.dt * self.nu * lap
            # # TODO implement random source?
            # if self.source != None:
            #     raise NotImplementedError("External source has not been implemented"
            #                               + " in global backprop.")

            new_state.append(real_state[i] + step)
            # shape broadcasting seems fine here, real_state shape=(1, 125), step shape=(125,), new_state shape=(1, 125)
        return tf.stack(new_state, axis=0)

    # TODO: modify this so less is in subclass? The reward section could go in the abstract class,
    # but this needs to be here because of how we calculate the WENO step. Not a high priority.
    # (And anyway, if we're refactoring, this should probably be the one canonical function instead
    # of having TF and non-TF versions.)
    # @tf.function
    def tf_calculate_reward(self, args):
        real_state, rl_state, rl_action, next_real_state = args
        # Note that real_state and next_real_state do not include ghost cells, but rl_state does.

        rl_state = rl_state[0] # Extract 1st (and only) dimension.
        rl_action = rl_action[0]

        # This section corresponds to envs/weno_solution.py#weno_weights_nd().
        C_values = weno_coefficients.C_all[self.weno_order]
        C_values = tf.constant(C_values, dtype=real_state.dtype)
        sigma_mat = weno_coefficients.sigma_all[self.weno_order]
        sigma_mat = tf.constant(sigma_mat, dtype=real_state.dtype)

        weno_action= []
        for i in range(self.grid.space.shape[0]):
            fp_stencils = rl_state[i, :, 0]
            fm_stencils = rl_state[i, :, 1]

            sub_stencil_indexes = create_stencil_indexes(stencil_size=self.weno_order,
                                                         num_stencils=self.weno_order)

            # Here axis 0 is location/batch and axis 1 is stencil,
            # so to index substencils we gather on axis 1.
            # Reverse because the constants expect that ordering (see weno_weights_nd()).
            sub_stencils_fp = tf.reverse(tf.gather(fp_stencils, sub_stencil_indexes, axis=1),
                                         axis=(-1,))  # For some reason, axis MUST be iterable, can't be scalar -1.
            sub_stencils_fm = tf.reverse(tf.gather(fm_stencils, sub_stencil_indexes, axis=1),
                                         axis=(-1,))

            q_squared_fp = sub_stencils_fp[:, :, None, :] * sub_stencils_fp[:, :, :, None]
            q_squared_fm = sub_stencils_fm[:, :, None, :] * sub_stencils_fm[:, :, :, None]

            beta_fp = tf.reduce_sum(sigma_mat * q_squared_fp, axis=(-2, -1))
            beta_fm = tf.reduce_sum(sigma_mat * q_squared_fm, axis=(-2, -1))

            epsilon = tf.constant(1e-16, dtype=tf.float64)
            alpha_fp = C_values / (epsilon + beta_fp ** 2)
            alpha_fm = C_values / (epsilon + beta_fm ** 2)

            weights_fp = alpha_fp / (tf.reduce_sum(alpha_fp, axis=-1)[:, None])
            weights_fm = alpha_fm / (tf.reduce_sum(alpha_fm, axis=-1)[:, None])
            weno_action.append(tf.stack([weights_fp, weights_fm], axis=1))
            # end adaptation of weno_weights_nd()
        weno_action = tf.stack(weno_action)

        weno_next_real_state = self.tf_integrate((real_state, (rl_state,), (weno_action,)))

        # This section is adapted from AbstactScalarEnv.calculate_reward()
        if "wenodiff" in self.reward_mode:
            last_action = rl_action

            action_diff = weno_action - last_action
            partially_flattened_shape = (self.action_space.shape[0],
                                         np.prod(self.action_space.shape[1:]))
            action_diff = tf.reshape(action_diff, partially_flattened_shape)
            if "L1" in self.reward_mode:
                error = tf.reduce_sum(tf.abs(action_diff), axis=-1)
            elif "L2" in self.reward_mode:
                error = tf.sqrt(tf.reduce_sum(action_diff ** 2, axis=-1))
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
                combined_error = tf.sqrt(tf.reduce_sum(error_stencils ** 2, axis=-1))
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

        #return reward
        return (reward,) # Singleton because this is 1D.
