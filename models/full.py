import os

import numpy as np
import tensorflow as tf
from tf.keras.layers import Layer

from util.misc import create_stencil_indexes
import envs.weno_coefficients as weno_coefficients

def makeRNN():
    policy = make_policy_network()
    cell = IntegrateCell(grid,
            prep_state_fn=WENO_prep_state,
            policy_net=policy,
            integrate_fn=WENO_integrate,
            reward_fn=WENO_reward,
            )
    rnn = IntegrateRNN(cell)

    states, actions, rewards = rnn(initial_state_ph, num_steps)

    loss = -tf.reduce_sum(rewards)



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
        #???
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
        # The input_size is irrelevant, right? It only ever depends on the initial state.
        # Build the cell.
        #???

        # Required by RNN API:
        self.state_size = None # Should be grid size. (Size of the recurrent state.)
        self.output_size = None # Should be size of the RL action.

        super().build()

    def call(state, **kwargs):

        real_state = state
        rl_state = self.prep_state_fn(real_state)

        rl_action = self.policy_net(rl_state)

        next_real_state = self.integrate_fn(real_state, rl_state, rl_action)

        rl_reward = self.reward_fn(real_state, rl_state, rl_action, next_real_state)

        return rl_action, rl_reward, next_real_state


def make_policy_network(state_size, action_size):
    #Make e.g. the 32,32 network.
    pass


"""
Based on prep_state in WENOBurgersEnv.
#TODO Create function in WENOBurgersEnv that returns this function?
"""
@tf.function
def WENO_prep_state(state):

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

"""
Based on a combination of functions in envs/burgers_env.py.
#TODO Same deal, could probably put this into the gym Env.
"""
@tf.function
def WENO_integrate(real_state, rl_state, rl_action):
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

"""
Based on a combination of functions in envs/burgers_env.py and in agents.py.
#TODO Same deal again, could probably put this into the Env.
"""
@tf.function
def WENO_reward(real_state, rl_state, rl_action, next_real_state):
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


