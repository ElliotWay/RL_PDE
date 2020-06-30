import numpy as np

import weno_coefficients


class StandardWENOAgent():
    def __init__(self, order=3):
        self.order = order

    def policy(self, state):
        #TODO: note that later the state will be formatted better as a batch,
        #i.e. (nx+1) X 2 X stencil_size instead of 2 X (nx+1) X stencil_size
        #which this assumes now.

        actions = self._weno_i_weights_batch(state)

        return actions

    def _weno_i_weights(self, q):
        """
        Get WENO weights at a given location in the grid
  
        Parameters
        ----------
        order : int
          The sub-stencil width.
        q : numpy array
          flux vector for that stencil.
  
        Returns
        -------
        Weights for sub-stencils in flux vector.
  
        """
        order = self.order

        C = weno_coefficients.C_all[order]
        sigma = weno_coefficients.sigma_all[order]
  
        beta = np.zeros((order))
        w = np.zeros_like(beta)
        epsilon = 1e-16
        alpha = np.zeros(order)
        for k in range(order):
            for l in range(order):
                for m in range(l+1):
                    beta[k] += sigma[k, l, m] * q[order-1+k-l] * q[order-1+k-m]
            alpha[k] = C[k] / (epsilon + beta[k]**2)
        w[:] = alpha / np.sum(alpha)
      
        return w

    def _weno_i_weights_batch(self, q_batch):
        """
        Get WENO weights for a batch
  
        Parameters
        ----------
        q_batch : numpy array
          Batch of flux stencils of size 2 (fp, fm) X grid length X stencil size
  
        Returns
        -------
        Weights for batch.
  
        """
        
        #TODO: vectorize properly
        weights_fp_stencil = []
        weights_fm_stencil = []
        batch_size = q_batch.shape[1]
        for i in range(batch_size):
            weights_fp_stencil.append(self._weno_i_weights(q_batch[0,i,:]))
            weights_fm_stencil.append(self._weno_i_weights(q_batch[1,i,:]))
          
        return np.array([weights_fp_stencil, weights_fm_stencil])
  
