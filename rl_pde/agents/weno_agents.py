import numpy as np

import envs.weno_coefficients as weno_coefficients
from envs.weno_solution import weno_weights_nd
from rl_pde.policy import Policy

# This could be a general ND agent easily enough,
# but we probably don't need any larger than a 2D agent.
class StandardWENO2DAgent(Policy):
    def __init__(self, order=3, mode="weno"):
        self.order = order
        self.mode = mode

    def predict(self, state, deterministic=False):
        weno_weights = tuple(weno_weights_nd(state_part) for state_part in state)

        if self.mode == "weno":
            return weno_weights, None
        elif self.mode == "split_flux":
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

        raise Exception("{} mode not implemented.".format(mode))

class StandardWENOAgent(Policy):
    def __init__(self, order=3, mode="weno"):
        self.order = order
        self.mode = mode

    def predict(self, state, deterministic=False):
        weno_weights = self._weno_weights_batch(state)

        if self.mode == "weno":
            return weno_weights, None

        if self.mode == "split_flux":
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

        raise Exception("{} mode not implemented.".format(mode))

    # Future note: we're wasting some computations.
    # The q^2 matrix for the kth sub-stencil has a lot of overlap with the q^2 matrix with the k+1st sub-stencil.
    # Also the upper triangle of the q^2 matrix is not needed.
    # Not a huge deal, but could be relevant with higher orders or factors of increasing complexity.
    def _weno_weights_batch(self, q_batch):
        """
        Get WENO weights for a batch
  
        Parameters
        ----------
        q_batch : numpy array
          Batch of flux stencils of size  grid length + 1 X 2 (fp, fm) X stencil size
  
        Returns
        -------
        Weights for batch.
  
        """

        order = self.order

        C = weno_coefficients.C_all[order]
        sigma = weno_coefficients.sigma_all[order]
        epsilon = 1e-16

        fp_stencils = q_batch[:, 0, :]
        fm_stencils = q_batch[:, 1, :]

        sliding_window_indexes = np.arange(order)[None, :] + np.arange(order)[:, None]

        # beta refers to the smoothness indicator.
        # To my best understanding, beta is calculated based on an approximation of the square of the values
        # in q, so we multiply every pair of values in a sub-stencil together, weighted appropriately for
        # the approximation by sigma.

        sub_stencils_fp = fp_stencils[:, sliding_window_indexes]
        sub_stencils_fm = fm_stencils[:, sliding_window_indexes]

        # Flipped because in original formulation we subtract from last index instead of adding from first,
        # and sigma and C weights assume this ordering.
        # Note: there was a typo in this line since before this file was created (July 2020) until
        # now (July 2021). Apparently this flip isn't important. I'm not sure why, but there's a
        # lot of symmetry going on - maybe there's a mathematical reason why they should be almost
        # the same.
        sub_stencils_fp = np.flip(sub_stencils_fp, axis=-1)
        sub_stencils_fm = np.flip(sub_stencils_fm, axis=-1)

        # Insert extra empty dimensions so numpy broadcasting produces
        # the desired outer product along only the intended dimensions.
        q_squared_fp = sub_stencils_fp[:, :, None, :] * sub_stencils_fp[:, :, :, None]
        q_squared_fm = sub_stencils_fm[:, :, None, :] * sub_stencils_fm[:, :, :, None]

        # Note that sigma is made up of triangular matrices, so the second copy of each pair is weighted by 0.
        beta_fp = np.sum(sigma * q_squared_fp, axis=(-2, -1))
        beta_fm = np.sum(sigma * q_squared_fm, axis=(-2, -1))

        alpha_fp = C / (epsilon + beta_fp ** 2)
        alpha_fm = C / (epsilon + beta_fm ** 2)

        # We need the [:, None] so numpy broadcasts to the sub-stencil dimension, instead of the stencil dimension.
        weights_fp = alpha_fp / (np.sum(alpha_fp, axis=-1)[:, None])
        weights_fm = alpha_fm / (np.sum(alpha_fm, axis=-1)[:, None])

        return np.stack([weights_fp, weights_fm], axis=1)
