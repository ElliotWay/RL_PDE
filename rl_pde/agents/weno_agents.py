import numpy as np

import envs.weno_coefficients as weno_coefficients
from envs.weno_solution import weno_weights_nd
from rl_pde.policy import Policy

# This could be a general ND agent easily enough,
# but we probably don't need any larger than a 2D agent.
class StandardWENO2DAgent(Policy):
    def __init__(self, order=3, action_type="weno"):
        self.order = order
        self.action_type = action_type

    def predict(self, state, deterministic=False):
        weno_weights = tuple(weno_weights_nd(state_part, self.order) for state_part in state)

        if self.action_type == "weno":
            return weno_weights, None
        elif self.action_type == "split_flux":
            raise Exception("I haven't tested this. This exception is safe to delete,"
                    + " but check that things are working as expected.")
            order = self.order
            a_mat = weno_coefficients.a_all[order]
            a_mat = np.flip(a_mat, axis=-1)
    
            # a_mat is [sub_stencil_index, inner_index]
            # weno_weights is [x, y, (fp,fm), sub_stencil_index]
            # We want [x, y, (fp,fm), inner_index].
            combined_weights = [a_mat[None, None, None, :, :] * weights[:, :, :, :, None]
                                        for weights in weno_weights]

            all_flux_weights = []
            for weights, state_part in zip(combined_weights, state):
                flux_weights = np.zeros_like(state_part)
                # TODO: figure out a way to vectorize this. This is a weird broadcast I'm trying to do,
                # which would be easier if numpy had builtin support for banded matrices.
                for sub_stencil_index in range(order):
                    flux_weights[:, :, sub_stencil_index:sub_stencil_index + order] += \
                                                            weights[:, :, sub_stencil_index, :]
            return all_flux_weights, None

        raise Exception("{} action type not implemented.".format(action_type))

class StandardWENOAgent(Policy):
    def __init__(self, order=3, action_type="weno"):
        print("Using standard WENO agent.")
        self.order = order
        self.action_type = action_type

    def predict(self, state, deterministic=False):
        weno_weights = weno_weights_nd(state, self.order)

        if self.action_type == "weno":
            return weno_weights, None

        if self.action_type == "split_flux":
            order = self.order
            a_mat = weno_coefficients.a_all[order]
            a_mat = np.flip(a_mat, axis=-1)
    
            # a_mat is [sub_stencil_index, inner_index]
            # weno_weights is [location, (fp,fm), sub_stencil_index]
            # We want [location, (fp,fm), inner_index].
            combined_weights = a_mat[None, None, :, :] * weno_weights[:, :, :, None]

            flux_weights = np.zeros_like(state)
            # TODO: figure out a way to vectorize this. This is a weird broadcast I'm trying to do,
            # which would be easier if numpy had builtin support for banded matrices.
            for sub_stencil_index in range(order):
                flux_weights[:, :, sub_stencil_index:sub_stencil_index + order] += combined_weights[:, :, sub_stencil_index, :]

            return flux_weights, None

        raise Exception("{} action type not implemented.".format(action_type))
