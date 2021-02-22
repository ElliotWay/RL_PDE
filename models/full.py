import os

import numpy as np
import tensorflow as tf
from tf.keras.layers import Layer

from util.misc import create_stencil_indexes
import envs.weno_coefficients as weno_coefficients

class IntegrateCell(Layer):
    def __init__(self, grid, prep_state_fn, policy_net, integrate_fn):
        super().__init__()
        self.grid = grid

        self.prep_state_fn = prep_state_fn
        self.policy_net = policy_net
        self.integrate_fn = integrate_fn
        
    def build(self, input_size):
        # The input_size is irrelevant, right? It only ever depends on the initial state.
        # Build the cell.
        #???

        # Required by RNN API:
        self.state_size = None # Should be grid size. (Size of the recurrent state.)
        self.output_size = None # Should be size of the RL action.

        super().build()

    def call(inputs, state, **kwargs):
        assert inputs is None, "inputs should be None - only needed as part of RNN API."

        # Convert input to RL state.
        real_state = state
        rl_state = self.prep_state_fn(real_state)

        # From the RL state, compute the RL action.
        rl_action = self.policy_net(rl_state)

        # From the RL state and the RL action, compute the next real state.
        next_real_state = self.integrate_fn(real_state, rl_state, rl_action)

        # Return the next real state AND the RL action (otherwise we never see the action).
        return rl_action, next_real_state

    def get_initial_state(inputs=None, batch_size=None, dtype=None):
        raise NotImplementedError("Integration must have initial state given explicitly.")

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

