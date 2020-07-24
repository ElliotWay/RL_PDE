import numpy as np

import envs.weno_coefficients as weno_coefficients


class StandardWENOAgent():
    def __init__(self, order=3):
        self.order = order

    def predict(self, state):
        # TODO: note that later the state will be formatted better as a batch,
        # i.e. (nx+1) X 2 X stencil_size instead of 2 X (nx+1) X stencil_size
        # which this assumes now.

        actions = self._weno_i_weights_batch(state)

        return actions, None

    # Future note: we're wasting some computations.
    # The q^2 matrix for the kth sub-stencil has a lot of overlap with the q^2 matrix with the k+1st sub-stencil.
    # Also the upper triangle of the q^2 matrix is not needed.
    # Not a huge deal, but could be relevant with higher orders or factors of increasing complexity.
    def _weno_i_weights_batch(self, q_batch):
        """
        Get WENO weights for a batch
  
        Parameters
        ----------
        q_batch : numpy array
          Batch of flux stencils of size 2 (fp, fm) X grid length + 1 X stencil size
  
        Returns
        -------
        Weights for batch.
  
        """

        order = self.order

        C = weno_coefficients.C_all[order]
        sigma = weno_coefficients.sigma_all[order]
        epsilon = 1e-16

        fp_stencils = q_batch[0]
        fm_stencils = q_batch[1]

        sliding_window_indexes = np.arange(order)[None, :] + np.arange(order)[:, None]

        # beta refers to the smoothness indicator.
        # To my best understanding, beta is calculated based on an approximation of the square of the values
        # in q, so we multiply every pair of values in a sub-stencil together, weighted appropriately for
        # the approximation by sigma.

        sub_stencils_fp = fp_stencils[:, sliding_window_indexes]
        sub_stencils_fm = fm_stencils[:, sliding_window_indexes]

        # Flipped because in original formulation we subtract from last index instead of adding from first,
        # and sigma and C weights assume this ordering.
        sub_stencilas_fp = np.flip(sub_stencils_fp, axis=-1)
        sub_stencilas_fm = np.flip(sub_stencils_fm, axis=-1)

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

        return np.array([weights_fp, weights_fm])



class StationaryAgent():
    """ Agent that always returns vectors of 0s, causing the environment to stay still. """

    def __init__(self, order=3):
        
        self.order = order

    def predict(self, state):

        state_shape = list(state.shape)
        state_shape[-1] = self.order
        action_shape = tuple(state_shape)

        return np.zeros(action_shape), None


class EqualAgent():
    """ Agent that always returns vectors of equal weight for each stencil. """

    def __init__(self, order=3):
        
        self.order = order

    def predict(self, state):

        action_shape = list(state.shape)
        action_shape[-1] = self.order
        action_shape = tuple(action_shape)

        return np.full(action_shape, 1.0 / self.order), None


class MiddleAgent():
    """ Agent that gives the middle stencil a weight of 1, and the rest 0. """

    def __init__(self, order=3):
        
        self.order = order

    def predict(self, state):

        action_shape = list(state.shape)
        action_shape[-1] = self.order
        action_shape = tuple(action_shape)

        middle = int(self.order / 2)

        weights = np.zeros(action_shape)
        weights[..., middle] = 1.0

        return weights, None


class LeftAgent():
    """ Agent that gives the leftmost stencil a weight of 1, and the rest 0. """

    def __init__(self, order=3):
        
        self.order = order

    def predict(self, state):

        action_shape = list(state.shape)
        action_shape[-1] = self.order
        action_shape = tuple(action_shape)

        middle = int(self.order / 2)

        weights = np.zeros(action_shape)
        weights[..., 0] = 1.0

        return weights, None


class RightAgent():
    """ Agent that gives the rightmost stencil a weight of 1, and the rest 0. """

    def __init__(self, order=3):
        
        self.order = order

    def predict(self, state):

        action_shape = list(state.shape)
        action_shape[-1] = self.order
        action_shape = tuple(action_shape)

        weights = np.zeros(action_shape)
        weights[..., -1] = 1.0

        return weights, None

