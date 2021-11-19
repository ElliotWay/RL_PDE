import numpy as np

from rl_pde.policy import Policy

# This could be extended to ND without too much difficulty.
class ExtendAgentVector(Policy):
    """
    Decorator that extends a 1-dimensional agent into 2 dimensions.

    The 1-dimensional agent is applied to every column and every row of a 2-dimensional
    environment.
    """
    def __init__(self, sub_agent):
        self.sub_agent = sub_agent

    def predict(self, state, deterministic=False):
        actions = []
        for index in range(state.shape[1]):
            state_slice = state[:, index]
            action, _ = self.sub_agent.predict(state_slice, deterministic=deterministic)
            actions.append(action)
        actions = np.stack(actions, axis=1)

        return actions, None
