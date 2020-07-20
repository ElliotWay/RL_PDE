import numpy as np

import weno_coefficients


class StationaryAgent():
    def __init__(self, order=3):
        self.order = order

    def predict(self, state):
        # TODO: note that later the state may be formatted better as a batch,
        # i.e. (nx+1) X 2 X stencil_size instead of 2 X (nx+1) X stencil_size
        # which this assumes now.

        state_shape = list(state.shape)
        state_shape[-1] = self.order
        action_shape = tuple(state_shape)

        return np.zeros(action_shape), _

