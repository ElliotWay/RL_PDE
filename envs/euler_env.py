import numpy as np
import tensorflow as tf
from gym import spaces

import envs.weno_coefficients as weno_coefficients
from envs.abstract_pde_env import AbstractPDEEnv
from envs.plottable_env import Plottable1DEnv
from envs.solutions import MemoizedSolution, OneStepSolution
from envs.solutions import RiemannSolution
from envs.weno_solution import WENOSolution, PreciseWENOSolution
from envs.weno_solution import weno_sub_stencils_nd, tf_weno_sub_stencils, tf_weno_weights
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
                self.solution = RiemannSolution(self.grid.nx, self.grid.ng, self.grid.xmin, self.grid.xmax,
                                                vec_len=3, init_type=kwargs['init_type'],
                                                gamma=kwargs['init_params']['eos_gamma'])
        else:
            if self.grid.ndim == 1:
                self.solution = PreciseWENOSolution(
                    self.grid, {'init_type': self.grid.init_type,
                                'boundary': self.grid.boundary},
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
                    self.grid, {'init_type': self.grid.init_type,
                                'boundary': self.grid.boundary},
                    precise_scale=1, precise_order=self.weno_order,
                    flux_function=self.euler_flux, source=self.source,
                    nu=nu, eqn_type='euler', vec_len=3)
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
        # epsilon = 1e-16
        flux = np.zeros_like(q)
        rho = q[0, :]
        S = q[1, :]
        E = q[2, :]
        v = S / rho
        # v = S / (rho + epsilon)
        p = (self.eos_gamma - 1) * (E - rho * v ** 2 / 2)
        flux[0, :] = S
        flux[1, :] = S * v + p
        flux[2, :] = (E + p) * v
        return flux

    def _rk_substep(self, action):
        rhs = super()._rk_substep(action)
        if self.nu > 0.0:
            rhs += self.nu * self.grid.laplacian()
        return rhs

    def _finish_step(self, step, dt, prev=None):
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

        # compute flux at each point
        f = self.euler_flux(g.u)
        # get maximum velocity
        alpha = self._max_lambda()

        # Lax Friedrichs Flux Splitting
        fp = (f + alpha * g.u) / 2
        fm = (f - alpha * g.u) / 2

        fp_stencil_indexes = create_stencil_indexes(stencil_size=self.state_order * 2 - 1,
                                                    num_stencils=g.real_length() + 1,
                                                    offset=g.ng - self.state_order)
        fm_stencil_indexes = fp_stencil_indexes + 1

        fp_stencils = [fp[i][fp_stencil_indexes] for i in range(g.space.shape[0])]
        fm_stencils = [fm[i][fm_stencil_indexes] for i in range(g.space.shape[0])]

        # Flip fm stencils. Not sure how I missed this originally?
        fm_stencils = np.flip(fm_stencils, axis=-1)
        fp_stencils = np.stack(fp_stencils, axis=0)

        # Stack fp and fm on axis -2 so grid position is still axis 0.
        self.current_state = np.stack([fp_stencils, fm_stencils], axis=-2)

        return self.current_state

    def _max_lambda(self):
        # epsilon = 1e-16
        rho = self.grid.u[0]
        v = self.grid.u[1] / rho
        # v = self.grid.u[1] / (rho + epsilon)
        p = (self.eos_gamma - 1) * (self.grid.u[2, :] - rho * v ** 2 / 2)
        # cs = np.sqrt(self.eos_gamma * p / (rho + epsilon))
        cs = np.sqrt(self.eos_gamma * p / rho)
        return max(np.abs(v) + cs)

    def _rk_substep(self, weights):

        state = self.current_state

        fp_stencils = state[:, :, 0, :]
        fm_stencils = state[:, :, 1, :]

        fp_weights = weights[:, :, 0, :]
        fm_weights = weights[:, :, 1, :]

        if self.reconstruction == 'componentwise':
            fp_substencils = weno_sub_stencils_nd(fp_stencils, self.weno_order)
            fm_substencils = weno_sub_stencils_nd(fm_stencils, self.weno_order)

            fpr = np.sum(fp_weights * fp_substencils, axis=-1)
            fml = np.sum(fm_weights * fm_substencils, axis=-1)
            flux = fml + fpr

        elif self.reconstruction == 'characteristicwise':
            boundary_state = (self.grid.u[:, self.grid.ng-1:-self.grid.ng] +
                              self.grid.u[:, self.grid.ng:-self.grid.ng+1]) * 0.5
            # boundary_state = (self.grid.u[:, self.grid.ng-1:-self.grid.ng] +
            #                   self.grid.u[:, self.grid.ng:-self.grid.ng+1]) * 0.5
            revecs, levecs = self._evecs_vectorized(boundary_state)
            # first index - primary variable vector (=3)
            # second index - stencil size
            # third index - grid
            char_fp = np.einsum('jli, jkl-> kli', fp_stencils, levecs)
            char_fm = np.einsum('jli, jkl-> kli', fm_stencils, levecs)

            fp_substencils = weno_sub_stencils_nd(char_fp, self.weno_order)
            fm_substencils = weno_sub_stencils_nd(char_fm, self.weno_order)

            fpr = np.sum(fp_weights * fp_substencils, axis=-1)
            fml = np.sum(fm_weights * fm_substencils, axis=-1)
            flux = np.einsum('jil, il-> jl', revecs, fpr + fml)

        else:
            raise NotImplementedError

        rhs = (flux[:, :-1] - flux[:, 1:]) / self.grid.dx

        rhs = rhs + super()._rk_substep(weights)

        return rhs

    def _evecs_vectorized(self, boundary_state):
        g = self.grid
        real_length = g.nx + 1
        revecs = np.zeros((3, 3, real_length))
        levecs = np.zeros((3, 3, real_length))
        rho = boundary_state[0, :]  # np.zeros((real_length))
        S = boundary_state[1, :]  # np.zeros((real_length))
        E = boundary_state[2, :]  # np.zeros((real_length))
        v = S / rho
        p = (self.eos_gamma - 1) * (E - rho * v ** 2 / 2)
        cs = np.sqrt(self.eos_gamma * p / rho)
        b1 = (self.eos_gamma - 1) / cs ** 2
        b2 = b1 * v ** 2 / 2
        revecs[0, 0, :] = 1
        revecs[0, 1, :] = v - cs
        revecs[0, 2, :] = (E + p) / rho - v * cs
        revecs[1, 0, :] = 1
        revecs[1, 1, :] = v
        revecs[1, 2, :] = v ** 2 / 2
        revecs[2, 0, :] = 1
        revecs[2, 1, :] = v + cs
        revecs[2, 2, :] = (E + p) / rho + v * cs
        levecs[0, 0, :] = (b2 + v / cs) / 2
        levecs[0, 1, :] = -(b1 * v + 1 / cs) / 2
        levecs[0, 2, :] = b1 / 2
        levecs[1, 0, :] = 1 - b2
        levecs[1, 1, :] = b1 * v
        levecs[1, 2, :] = -b1
        levecs[2, 0, :] = (b2 - v / cs) / 2
        levecs[2, 1, :] = -(b1 * v - 1 / cs) / 2
        levecs[2, 2, :] = b1 / 2
        return revecs, levecs

    def tf_revecs_vectorized(self, boundary_state):
        data_type = boundary_state.dtype
        rho = boundary_state[0, :]  # np.zeros((real_length))
        S = boundary_state[1, :]  # np.zeros((real_length))
        E = boundary_state[2, :]  # np.zeros((real_length))
        v = S / rho
        eos_gamma = tf.convert_to_tensor(self.eos_gamma, dtype=data_type)
        p = (eos_gamma - 1) * (E - rho * v ** 2 / 2)
        cs = tf.sqrt(eos_gamma * p / rho)
        ones = tf.ones(boundary_state.shape[-1], dtype=data_type)
        revecs = tf.stack([tf.stack([ones, v - cs, (E + p) / rho - v * cs]),
                           tf.stack([ones, v, v ** 2 / 2]),
                           tf.stack([ones, v + cs, (E + p) / rho + v * cs])
                           ])
        return revecs

    def tf_levecs_vectorized(self, boundary_state):
        data_type = boundary_state.dtype
        rho = boundary_state[0, :]  # np.zeros((real_length))
        S = boundary_state[1, :]  # np.zeros((real_length))
        E = boundary_state[2, :]  # np.zeros((real_length))
        v = S / rho
        eos_gamma = tf.convert_to_tensor(self.eos_gamma, dtype=data_type)
        p = (eos_gamma - 1) * (E - rho * v ** 2 / 2)
        cs = tf.sqrt(eos_gamma * p / rho)
        b1 = (eos_gamma - 1) / cs ** 2
        b2 = b1 * v ** 2 / 2
        ones = tf.ones(boundary_state.shape[-1], dtype=data_type)
        levecs = tf.stack([tf.stack([(b2 + v / cs) / 2, -(b1 * v + ones / cs) / 2, b1 / 2]),
                           tf.stack([ones - b2, b1 * v, -b1]),
                           tf.stack([(b2 - v / cs) / 2, -(b1 * v - ones / cs) / 2, b1 / 2])
                           ])
        return levecs

    # @tf.function
    def tf_prep_state(self, state):
        # Note the similarity to prep_state() above.

        # state (the real physical state) does not have ghost cells, but agents operate on a stencil
        # that can spill beyond the boundary, so we need to add ghost cells to create the rl_state.

        boundary = self.grid.boundary
        full_state = self.grid.tf_update_boundary(state, boundary)

        # Compute flux. Euler specific!!!
        rho = full_state[0, :]
        S = full_state[1, :]
        E = full_state[2, :]
        v = S / rho
        p = (self.eos_gamma - 1) * (E - rho * v ** 2 / 2)
        flux = tf.stack([S, S * v + p, (E + p) * v], axis=0)

        cs = tf.sqrt(self.eos_gamma * p / rho)
        alpha = tf.reduce_max(tf.abs(v) + cs)

        flux_plus = (flux + alpha * full_state) / 2
        flux_minus = (flux - alpha * full_state) / 2

        plus_indexes = create_stencil_indexes(
            stencil_size=self.state_order * 2 - 1,
            num_stencils=self.nx + 1,
            offset=self.ng - self.state_order)
        minus_indexes = np.flip(plus_indexes, axis=-1) + 1

        plus_stencils = [tf.gather(flux_plus[i], plus_indexes) for i in range(self.grid.space.shape[0])]
        minus_stencils = [tf.gather(flux_minus[i], minus_indexes) for i in range(self.grid.space.shape[0])]

        # Stack together into rl_state.
        # Stack on dim 2 to keep location dim 1. (different from Burgers, 0th dim is vec_length)
        rl_state = tf.stack([plus_stencils, minus_stencils], axis=2)

        if self.reconstruction =='componentwise':
            return (rl_state,)  # second item not used, but have to return two items for TF graph

        if self.reconstruction == 'characteristicwise':
            boundary = self.grid.boundary
            if not type(boundary) is str:
                boundary = boundary[0]

            if boundary == "outflow":
                left_state = tf.repeat(tf.expand_dims(state[:, 0], 1), self.ng, axis=1)
                right_state = tf.repeat(tf.expand_dims(state[:, -1], 1), self.ng, axis=1)
            elif boundary == "periodic":
                left_state = state[:, -self.ng:]
                right_state = state[:, :self.ng]
            else:
                raise NotImplementedError()
            full_state = tf.concat([left_state, state, right_state], axis=1)

            boundary_state = (full_state[:, self.ng - 1:-self.ng] + full_state[:, self.ng:-self.ng + 1]) * 0.5

            levecs = self.tf_levecs_vectorized(boundary_state)
            char_fp = tf.einsum('jli, jkl-> kli', rl_state[:, :, 0], levecs)
            char_fm = tf.einsum('jli, jkl-> kli', rl_state[:, :, 1], levecs)
            rl_state = tf.stack([char_fp, char_fm], axis=2)
            return (rl_state,)

    # @tf.function
    def tf_integrate(self, args):
        real_state, rl_state, rl_action = args

        rl_state = rl_state[0]  # Extract 1st (and only) dimension.
        rl_action = rl_action[0]

        # Note that real_state does not contain ghost cells here, but rl_state DOES (and rl_action has
        # weights corresponding to the ghost cells).

        plus_stencils = rl_state[:, :, 0]
        minus_stencils = rl_state[:, :, 1]
        plus_action = rl_action[:, :, 0]
        minus_action = rl_action[:, :, 1]

        fpr = tf.reduce_sum(plus_action * tf_weno_sub_stencils(plus_stencils, self.weno_order), axis=-1)
        fml = tf.reduce_sum(minus_action * tf_weno_sub_stencils(minus_stencils, self.weno_order), axis=-1)

        if self.reconstruction == 'componentwise':
            reconstructed_flux = fpr + fml

        elif self.reconstruction == 'characteristicwise':
            boundary = self.grid.boundary
            if not type(boundary) is str:
                boundary = boundary[0]

            if boundary == "outflow":
                left_state = tf.repeat(tf.expand_dims(real_state[:, 0], 1), self.ng, axis=1)
                right_state = tf.repeat(tf.expand_dims(real_state[:, -1], 1), self.ng, axis=1)
            elif boundary == "periodic":
                left_state = real_state[:, -self.ng:]
                right_state = real_state[:, :self.ng]
            else:
                raise NotImplementedError()
            full_state = tf.concat([left_state, real_state, right_state], axis=1)

            boundary_state = (full_state[:, self.ng - 1:-self.ng] + full_state[:, self.ng:-self.ng + 1]) * 0.5

            revecs = self.tf_revecs_vectorized(boundary_state)
            reconstructed_flux = tf.einsum('jil, il-> jl', revecs, fpr + fml)

        else:
            raise NotImplementedError

        derivative_u_t = (reconstructed_flux[:, :-1] - reconstructed_flux[:, 1:]) / self.grid.dx

        # TODO implement RK4?

        dt = self.tf_timestep(real_state)

        step = dt * derivative_u_t

        if self.nu != 0.0:
            step += dt * self.nu * self.grid.tf_laplacian(real_state)
        # TODO implement random source?
        if self.source != None:
            raise NotImplementedError("External source has not been implemented"
                                      + " in global backprop.")

        new_state = real_state + step

        return new_state

    # TODO: modify this so less is in subclass? The reward section could go in the abstract class,
    # but this needs to be here because of how we calculate the WENO step. Not a high priority.
    # (And anyway, if we're refactoring, this should probably be the one canonical function instead
    # of having TF and non-TF versions.)
    # @tf.function
    def tf_calculate_reward(self, args):
        real_state, rl_state, rl_action, next_real_state = args
        # Note that real_state and next_real_state do not include ghost cells, but rl_state does.

        rl_state = rl_state[0]  # Extract 1st (and only) dimension.
        rl_action = rl_action[0]

        fp_stencils = rl_state[:, :, 0]
        fm_stencils = rl_state[:, :, 1]

        fp_weights = tf_weno_weights(fp_stencils, self.weno_order)
        fm_weights = tf_weno_weights(fm_stencils, self.weno_order)
        weno_action = tf.stack([fp_weights, fm_weights], axis=-2)

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

        # return reward
        return (reward,)  # Singleton because this is 1D.
