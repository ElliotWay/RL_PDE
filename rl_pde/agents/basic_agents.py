import numpy as np

from rl_pde.policy import Policy

class StationaryAgent(Policy):
    """ Agent that always returns vectors of 0s, causing the environment to stay still. """

    def __init__(self, order=3, action_type="weno"):
        
        self.order = order
        self.action_type = action_type

    def predict(self, state, deterministic=False):

        if self.action_type == "weno":
            state_shape = list(state.shape)
            state_shape[-1] = self.order
            action_shape = tuple(state_shape)
        elif self.action_type == "split_flux" or self.action_type == "flux":
            action_shape = state.shape

        return np.zeros(action_shape), None


class EqualAgent(Policy):
    """ Agent that always returns vectors of equal weight for each stencil. """

    def __init__(self, order=3):
        
        self.order = order

    def predict(self, state, deterministic=False):

        action_shape = list(state.shape)
        action_shape[-1] = self.order
        action_shape = tuple(action_shape)

        return np.full(action_shape, 1.0 / self.order), None


class MiddleAgent(Policy):
    """ Agent that gives the middle stencil a weight of 1, and the rest 0. """

    def __init__(self, order=3):
        
        self.order = order

    def predict(self, state, deterministic=False):

        action_shape = list(state.shape)
        action_shape[-1] = self.order
        action_shape = tuple(action_shape)

        middle = int(self.order / 2)

        weights = np.zeros(action_shape)
        weights[..., middle] = 1.0

        return weights, None


class LeftAgent(Policy):
    """ Agent that gives the leftmost stencil a weight of 1, and the rest 0. """

    def __init__(self, order=3):
        
        self.order = order

    def predict(self, state, deterministic=False):

        action_shape = list(state.shape)
        action_shape[-1] = self.order
        action_shape = tuple(action_shape)

        middle = int(self.order / 2)

        weights = np.zeros(action_shape)
        weights[..., 0] = 1.0

        return weights, None


class RightAgent(Policy):
    """ Agent that gives the rightmost stencil a weight of 1, and the rest 0. """

    def __init__(self, order=3):
        
        self.order = order

    def predict(self, state, deterministic=False):

        action_shape = list(state.shape)
        action_shape[-1] = self.order
        action_shape = tuple(action_shape)

        weights = np.zeros(action_shape)
        weights[..., -1] = 1.0

        return weights, None

class RandomAgent(Policy):
    """ Agent that gives random weights (that still add up to 1). """

    def __init__(self, order=3, action_type="weno"):

        self.order = order
        self.action_type = action_type

    def predict(self, state, deterministic=False):
        # deterministic argument is ignored - this class is meant to represent a random policy,
        # not a policy with random actions.

        if self.action_type == "weno":
            action_shape = list(state.shape)
            action_shape[-1] = self.order
            action_shape = tuple(action_shape)

            # Do Gaussian sample, then apply softmax.
            random_logits = np.random.normal(size=action_shape)
            exp_logits = np.exp(random_logits)

            action = exp_logits / (np.sum(exp_logits, axis=-1)[..., None])
            return action, None
        elif self.action_type == "split_flux" or self.action_type == "flux": 
            # e^(order - 1) is chosen ad-hoc to vaguely relate to the max weights in WENO that increase with order.
            return np.random.normal(size=state.shape, scale=(np.exp(self.order - 1))), None

        raise Exception("{} action type not implemented.".format(action_type))
