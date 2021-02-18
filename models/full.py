import os

import numpy as np
import tensorflow as tf
from tf.keras.layers import Layer

from util.misc import create_stencil_indexes

class IntegrateCell(Layer):
    def __init__(self, grid, prep_state_fn, policy_layer, integrate_fn):
        super().__init__()
        self.grid = grid

        self.prep_state_fn = prep_state_fn
        self.policy_layer = policy_layer
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
        rl_action = self.policy_layer(rl_state)

        # From the RL state and the RL action, compute the next real state.
        next_real_state = self.integrate_fn(rl_state, rl_action)

        # Return the next real state AND the RL action (otherwise we never see the action).
        return rl_action, next_real_state

    def get_initial_state(inputs=None, batch_size=None, dtype=None):
        raise NotImplementedError("Integration must have initial state given explicitly.")


"""
Based on prep_state in WENOBurgersEnv.
#TODO Create function in WENOBurgersEnv that returns this function?
"""
@tf.function
def WENO_prep_state(state):

    # Expand boundaries? Or is that already in state?

    # Compute flux.
    flux = 0.5 * (state ** 2)

    alpha = tf.reduce_max(tf.abs(flux))

    flux_plus = (flux + alpha * state) / 2
    flux_minus = (flux - alpha * state) / 2

    plus_indexes = create_stencil_indexes(
                    stencil_size=(self.weno_order * 2 - 1),
                    num_stencils=(self.nx + 1),
                    offset=(self.ng - self.weno_order))
    minus_indexes = plus_index + 1
    minus_indexes = np.flip(minus_index, axis=#??

    fp_stencils = flux_plus[plus_indexes]
    fm_stencils = flux_minus[minus_indexes]

    #Stack together into rl_state.

    return rl_state
