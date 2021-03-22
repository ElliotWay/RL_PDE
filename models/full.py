import os

import numpy as np
import tensorflow as tf
from tf.keras.layers import Layer

from models.builder import get_optimizer
from models.net import PolicyNet
from util.misc import create_stencil_indexes
from util.serialize import save_to_zip, load_from_zip
import envs.weno_coefficients as weno_coefficients

# Unlike our other models, it doesn't make sense to train with a set of trajectories, as the global
# model has trajectories as part of its internal workings.
# Instead, the only training input is the initial state, so the train function is changed to
# train(initial_state) instead.
# The predict method, on the other hand, is the same, as the global must still have the local model
# built in.
class GlobalBackpropModel:
    def __init__(self, env, args, dtype=tf.float32):
        self.dtype = dtype
        self.env = env

        self.session = tf.Session()
        

        #TODO Use ReLU? A CL argument with ReLU as the default might make sense.
        with tf.variable_scope("policy", reuse=False):
            self.policy = PolicyNet(layers=args.layers, action_shape=env.action_space.shape,
                    activation_fn=tf.nn.relu)
        # Direct policy input and output used in predict() method during inference.
        self.policy_input_ph = tf.placeholder(dtype=dtype,
                shape=(None,) + env.observation_space.shape[1:], name="policy_input")
        self.policy_output = self.policy(self.policy_input_ph)

        cell = IntegrateCell(grid,
                prep_state_fn=WENO_prep_state,
                policy_net=self.policy,
                integrate_fn=WENO_integrate,
                reward_fn=WENO_reward,
                )
        rnn = IntegrateRNN(cell)

        # initial_state_ph is the REAL physical initial state
        #TODO should initial_state_ph include ghost cells?
        self.initial_state_ph = tf.placeholder(dtype=dtype,
                shape=(None, args.nx), name="init_real_state")

        #TODO possibly restrict num_steps to something smaller?
        # We'd then need to sample timesteps from a WENO trajectory.

        states, actions, rewards = rnn(initial_state_ph, num_steps=args.ep_length)

        assert tf.rank(rewards) == 3
        # Sum over the length of the trajectory, then average over each location,
        # and average over the batch.
        self.loss = -tf.reduce_mean(tf.reduce_sum(rewards, axis=2))

        params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="policy")
        print("{} params - does that look right compared to layers={} ?".format(
            len(params), layers)
        gradients = tf.gradients(self.loss, params)
        grads = list(zip(gradients, params))
        self.params = params

        self.optimizer = get_optimizer(args)
        self.train_policy = self.optimizer.apply_gradients(grads)

        #TODO do something with states/actions as a debug output?

    def train(self, initial_state):

        feed_dict = {self.initial_state_ph:initial_state}

        loss, _ = self.session.run([self.loss, self.train_policy], feed_dict=feed_dict)
        return {"loss":loss}

    def predict(self, state, deterministic=True):

        single_obs_rank = len(self.env.observation_space.shape) - 1
        input_rank = len(state.shape)

        if input_rank == single_obs_rank:
            # Single local state.
            assert state.shape == self.env.observation_space.shape[1:]
            state = state[None]
            action = self.session.run(self.policy_output, feed_dict={self.policy_input_ph:state})
            action = action[0]

        elif input_rank == single_obs_rank + 1:
            # Batch of local states 
            # OR single global state.
            assert state.shape[1:] == self.env.observation_space.shape[1:]
            action = self.session.run(self.policy_output, feed_dict={self.policy_input_ph:state})

        elif input_rank == single_obs_rank + 2:
            # Batch of global states.
            assert state.shape[1:] == self.env.observation_space.shape
            batch_length = state.shape[0]
            spatial_length = state.shape[1]
            flattened_state = state.reshape((batch_length * spatial_length,) + state.shape[2:])
            flattened_action = self.session.run(self.policy_output,
                    feed_dict={self.policy_input_ph:flattened_state})
            action = flattened_action.reshape((batch_length, spatial_length,) + action.shape[1:])

        return action

    def save(self, path):
        #return path
        raise NotImplementedError

    def load(self, path):
        raise NotImplementedError


class IntegrateRNN(Layer):
    def __init__(self, cell, dtype=tf.float32, swap_to_cpu=True):
        """
        Declare the RNN for PDE integration.
        
        Parameters
        ----------
        cell : tf.keras.layers.Layer
          The recurrent cell. Probably an IntegrateCell.
          Needs to have a call(previous_state) method returning (action, next_state).
        dtype : dtype
          Type of the state (probably tf.float32 or tf.float64).
        swap_to_cpu : bool
          Whether to copy Tensors to the CPU from the GPU. Used by the tf.while_loop() function.
          With long chains, too many Tensors on the GPU will exhaust the memory, but if the chain
          is relatively short (how short depends on the GPU), you can (probably) increase
          performance by keeping the Tensors on the GPU instead. True by default. Does nothing if
          only using the CPU anyway.
        """
        super().__init__()
        self.cell = cell
        self.dtype = dtype
        self.swap_to_cpu = swap_to_cpu

    def build(self, input_size):
        super().build()

    def call(initial_state, num_steps):
        states_ta = tf.TensorArray(dtype=self.dtype, size=num_steps)
        actions_ta = tf.TensorArray(dtype=self.dtype, size=num_steps)
        rewards_ta = tf.TensorArray(dtype=self.dtype, size=num_steps)

        def condition_fn(time, *_):
            return time < num_steps

        def loop_fn(time, states_ta, actions_ta, rewards_ta, current_state):
            next_action, next_reward, next_state = self.cell(current_state)

            # Overwriting the TensorArrays is necessary to keep connections
            # through the entire network graph.
            states_ta = states_ta.write(time, next_state)
            actions_ta = actions_ta.write(time, next_action)
            rewards_ta = rewards_ta.write(time, next_reward)

            return (time + 1, states_ta, actions_ta, rewards_ta, next_state)

        initial_time = tf.Constant(0, dtype=tf.int32)

        _time, states_ta, actions_ta, rewards_ta, final_state = tf.while_loop(
                cond=condition_fn,
                body=loop_fn,
                loop_vars=(initial_time, states_ta, actions_ta, rewards_ta, initial_state),
                parallel_iterations=10, # Iterations of the loop to run in parallel?
                                        # I'm not sure how that works.
                                        # 10 is the default value. Keras backend uses 32.
                swap_memory=self.swap_to_cpu,
                return_same_structure=True)

        states = states_ta.stack()
        actions = actions_ta.stack()
        rewards = rewards_ta.stack()
        return states, actions, rewards


class IntegrateCell(Layer):
    def __init__(self, grid, prep_state_fn, policy_net, integrate_fn, reward_fn):
        super().__init__()
        self.grid = grid

        self.prep_state_fn = prep_state_fn
        self.policy_net = policy_net
        self.integrate_fn = integrate_fn
        self.reward_fn = reward_fn
        
    def build(self, input_size):
        super().build()

    def call(self, state):
        # state has shape [batch, location, ...]

        real_state = state
        # Use tf.map_fn to apply function across every element in the batch.
        rl_state = tf.map_fn(self.prep_state_fn, real_state)

        # The policy expects a batch so we don't need to use tf.map_fn;
        # however, it expects batches of INDIVIDUAL states, not every location at once, so we need
        # to combine the batch and location axes first (and then reshape the actions back).
        rl_state_shape = rl_state.get_shape().as_list()
        new_state_shape = [rl_state_shape[0] + rl_state_shape[1],] + rl_state_shape[2:]
        reshaped_state = tf.reshape(rl_state, new_state_shape)

        shaped_action = self.policy_net(reshaped_state)

        shaped_action_shape = shaped_action.get_shape().as_list()
        rl_action_shape = [rl_state_shape[0], rl_state_shape[1]] + shaped_action_shape[1:]
        rl_action = tf.reshape(shaped_action, rl_action_shape)

        next_real_state = tf.map_fn(self.integrate_fn, (real_state, rl_state, rl_action))

        rl_reward = tf.map_fn(self.reward_fn, (real_state, rl_state, rl_action, next_real_state))

        return rl_action, rl_reward, next_real_state


@tf.function
def WENO_prep_state(state):
    """
    Based on prep_state in WENOBurgersEnv.
    #TODO Create function in WENOBurgersEnv that returns this function?
    """

    #TODO Expand boundaries? Or is that already in state?

    # Compute flux.
    flux = 0.5 * (state ** 2)

    alpha = tf.reduce_max(tf.abs(flux))

    flux_plus = (flux + alpha * state) / 2
    flux_minus = (flux - alpha * state) / 2

    #TODO Could change things to use traditional convolutions instead.
    # Maybe if this whole thing actually works.

    plus_indexes = create_stencil_indexes(
                    stencil_size=(self.weno_order * 2 - 1),
                    num_stencils=(self.nx + 1),
                    offset=(self.ng - self.weno_order))
    minus_indexes = plus_index + 1
    minus_indexes = np.flip(minus_indexes, axis=-1)

    plus_stencils = flux_plus[plus_indexes]
    minus_stencils = flux_minus[minus_indexes]

    # Stack together into rl_state.
    # Stack on dim 1 to keep location dim 0.
    rl_state = tf.stack([plus_stencils, minus_stencils], axis=1)

    return rl_state

@tf.function
def WENO_integrate(args):
    """
    Based on a combination of functions in envs/burgers_env.py.
    #TODO Same deal, could probably put this into the gym Env.

    Parameters
    ----------
    args : tuple
        The tuple should be made up of (real_state, rl_state, rl_action).
        (Needs to be a tuple so this function works with tf.map_fn.)
    """
    real_state, rl_state, rl_action = args

    # TODO does real_state contain ghost cells here?
    # Don't forget that both the cell and the reward function call this.

    plus_stencils = rl_state[:, 0]
    minus_stencils = rl_state[:, 1]

    #weno_i_stencils_batch()
    a_mat = weno_coefficients.a_all[self.order]
    a_mat = np.flip(a_mat, axis=-1)
    a_mat = tf.constant(a_mat)
    sub_stencil_indexes = create_stencil_indexes(stencil_size=self.order, num_stencils=self.order)
    plus_interpolated = tf.reduce_sum(a_mat * plus_stencils[:, sub_stencil_indexes], axis=-1)
    minus_interpolated = tf.reduce_sum(a_mat * minus_stencils[:, sub_stencil_indexes], axis=-1)

    plus_action = rl_action[:, 0]
    minus_action = rl_action[:, 1]

    fpr = tf.reduce_sum(plus_action * plus_interpolated, axis=-1)
    fml = tf.reduce_sum(minus_action * minus_interpolated, axis=-1)

    reconstructed_flux = fpr + fml

    derivative_u_t = (reconstructed_flux[:-1] - reconstructed_flux[1:]) / self.dx

    #TODO implement RK4 as well?

    step = self.dt * derivative_u_t

    #TODO implement viscosity and random source?
    
    new_state = real_state + step
    return new_state

@tf.function
def WENO_reward(args):
    """
    Based on a combination of functions in envs/burgers_env.py and in agents.py.
    #TODO Same deal again, could probably put this into the Env.

    Parameters
    ----------
    args : tuple
        The tuple should be made up of (real_state, rl_state, rl_action, next_real_state).
        (Needs to be a tuple so this function works with tf.map_fn.)
    """
    real_state, rl_state, rl_action, next_real_state = args
    # Note that real_state and next_real_state do not include ghost cells.

    # agents.py#StandardWENOAgent._weno_weights_batch()
    C_values = weno_coefficients.C_all[self.order]
    C_values = tf.constant(C_values)
    sigma_mat = weno_coefficients.sigma_all[self.order]
    sigma_mat = tf.constant(sigma_mat)

    fp_stencils = rl_state[:, 0]
    fm_stencils = rl_state[:, 1]

    sub_stencil_indexes = create_stencil_indexes(stencil_size=self.order, num_stencils=self.order)

    sub_stencils_fp = tf.reverse(fp_stencils[:, sub_stencil_indexes], axis=-1)
    sub_stencils_fm = tf.reverse(fm_stencils[:, sub_stencil_indexes], axis=-1)

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

    weno_next_real_state = WENO_integrate(real_state, rl_state, WENO_action)

    #envs/burgers_env.py#AbstactBurgersEnv.calculate_reward()
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
            raise Exception("AbstractBurgersEnv: reward_mode problem")

        return -error

    error = self.solution.get_full() - self.grid.get_full()

    if "one-step" in self.reward_mode:
        # This version is ALWAYS one-step - the others are tricky to implement in TF.
    elif "full" in self.reward_mode:
        raise Exception("Reward mode 'full' invalid - only 'one-step' reward implemented"
            " in Tensorflow functions.")
    elif "change" in self.reward_mode:
        raise Exception("Reward mode 'change' invalid - only 'one-step' reward implemented"
            " in Tensorflow functions.")
    else:
        raise Exception("AbstractBurgersEnv: reward_mode problem")

    if "clip" in self.reward_mode:
        raise Exception("Reward clipping not implemented in Tensorflow functions.")

    # Average of error in two adjacent cells.
    if "adjacent" in self.reward_mode and "avg" in self.reward_mode:
        error = tf.abs(error)
        combined_error = (error[:-1] + error[1:]) / 2
        # Assume error beyond boundaries is 0, so error at edge interfaces is error at edge cells.
        combined_error = tf.concat([error[0] / 2, combined_error, error[-1] / 2], axis=0)
    # Combine error across the stencil.
    elif "stencil" in self.reward_mode:
        # This is trickier and we need to expand the error into the ghost cells.
        # Still assume the error is 0 for convenience.
        ghost_error = tf.constant([0.0] * self.ng)
        full_error = tf.concat([ghost_error, error, ghost_error])
        
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
            raise Exception("AbstractBurgersEnv: reward_mode problem")
    else:
        raise Exception("AbstractBurgersEnv: reward_mode problem")

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
            reward_adjustment = tf.contant(self.reward_adjustment)
            # The constant controls the relative importance of small rewards compared to large rewards.
            # Towards infinity, all rewards (or penalties) are equally important.
            # Towards 0, small rewards are increasingly less important.
            # An alternative to arctan(C*x) with this property would be x^(1/C).
            reward = tf.atan(reward_adjustment * -combined_error)
    else:
        raise Exception("AbstractBurgersEnv: reward_mode problem")

    # end adaptation of calculate_reward()

    return reward


