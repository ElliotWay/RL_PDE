import gym
import numpy as np
import tensorflow as tf
from gym import spaces

from envs.abstract_scalar_env import AbstractScalarEnv
from envs.plottable_env import Plottable1DEnv
from envs.weno_solution import WENOSolution, PreciseWENOSolution, PreciseWENOSolution2D
from envs.solutions import AnalyticalSolution
from envs.solutions import MemoizedSolution, OneStepSolution
import envs.weno_coefficients as weno_coefficients
from util.softmax_box import SoftmaxBox
from util.misc import create_stencil_indexes

#TODO Have a common declaration for these - in weno_solution.py perhaps?
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

class AbstractBurgersEnv(AbstractScalarEnv):
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
        *args, **kwargs
            The remaining arguments are passed to AbstractScalarEnv.
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
                        self.grid.xmin, self.grid.xmax, vec_len=1, init_type=init_type)
        else:
            if self.grid.ndim == 1:
                self.solution = PreciseWENOSolution(
                        self.grid, {'init_type':self.grid.init_type,
                            'boundary':self.grid.boundary},
                        precise_scale=precise_scale, precise_order=precise_weno_order,
                        flux_function=self.burgers_flux, source=self.source,
                        nu=nu, vec_len=1)
            elif self.grid.ndim == 2:
                self.solution = PreciseWENOSolution2D(
                        self.grid, {'init_type':self.grid.init_type,
                            'boundary':self.grid.boundary},
                        precise_scale=precise_scale, precise_order=precise_weno_order,
                        flux_function=self.burgers_flux, source=self.source,
                        nu=nu, vec_len=1)
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
                        flux_function=self.burgers_flux, source=self.source,
                        nu=nu, vec_len=1)
            elif self.grid.ndim == 2:
                self.weno_solution = PreciseWENOSolution2D(
                        self.grid, {'init_type':self.grid.init_type,
                            'boundary':self.grid.boundary},
                        precise_scale=1, precise_order=self.weno_order,
                        flux_function=self.burgers_flux, source=self.source,
                        nu=nu, vec_len=1)

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

    def _finish_step(self, step, dt, prev=None):
        if self.nu > 0.0:
            R = self.nu * self.grid.laplacian()
            step += dt * R

        state, reward, done = super()._finish_step(step, dt, prev)

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

        self._action_labels = ["$w^{}_{}$".format(sign, num) for sign in ['+', '-']
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

    def _rk_substep(self, weights):

        state = self.current_state

        fp_state = state[:, 0, :]
        fm_state = state[:, 1, :]

        #TODO state_order != weno_order has never worked well.
        # Is this why? Should this be state order? Or possibly it should be weno order but we still
        # need to compensate for a larger state order somehow?
        fp_stencils = weno_i_stencils_batch(self.weno_order, fp_state)
        fm_stencils = weno_i_stencils_batch(self.weno_order, fm_state)

        fp_weights = weights[:, 0, :]
        fm_weights = weights[:, 1, :]

        fpr = np.sum(fp_weights * fp_stencils, axis=-1)
        fml = np.sum(fm_weights * fm_stencils, axis=-1)

        flux = fml + fpr

        rhs = (flux[:-1] - flux[1:]) / self.grid.dx

        return rhs

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
            raise NotImplementedError("{} boundary not implemented.".format(self.grid.boundary))

        # Compute flux. Burgers specific!!!
        flux = 0.5 * (full_state ** 2)

        alpha = tf.reduce_max(tf.abs(flux))

        flux_plus = (flux + alpha * full_state) / 2
        flux_minus = (flux - alpha * full_state) / 2

        #TODO Could change things to use traditional convolutions instead.
        # Maybe if this whole thing actually works.

        plus_indexes = create_stencil_indexes(
                        stencil_size=self.state_order * 2 - 1,
                        num_stencils=self.nx + 1,
                        offset=self.ng - self.state_order)
        #plus_indexes = create_stencil_indexes(
                        #stencil_size=(self.weno_order * 2 - 1),
                        #num_stencils=(self.nx + 1),
                        #offset=(self.ng - self.weno_order))
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

        if self.nu != 0.0:
            # Compute the Laplacian. This involves the first ghost cell past the boundary.
            central_lap = (real_state[:-2] - 2.0*real_state[1:-1] + real_state[2:]) / (self.grid.dx**2)
            if self.grid.boundary == "outflow":
                left_lap = (-real_state[0] + real_state[1]) / (self.grid.dx**2) # X-2X+Y = -X+Y
                right_lap = (real_state[-2] - real_state[-1]) / (self.grid.dx**2) # X-2Y+Y = X-Y
            elif self.grid.boundary == "periodic":
                left_lap = (real_state[-1] - 2.0*real_state[0] + real_state[1]) / (self.grid.dx**2)
                right_lap = (real_state[-2] - 2.0*real_state[-1] + real_state[0]) / (self.grid.dx**2)
            else:
                raise NotImplementedError()
            lap = tf.concat([[left_lap], central_lap, [right_lap]], axis=0)

            step += self.dt * self.nu * lap
        #TODO implement random source?
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

        epsilon = tf.constant(1e-16, dtype=tf.float64)
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
                                       dtype=np.float32)
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
                                       dtype=np.float32)
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

    def _rk_step(self, action):
        state = self.current_state
        flux = np.sum(action * state, axis=-1)
        rhs = (flux[:-1] - flux[1:]) / self.grid.dx

        return rhs
