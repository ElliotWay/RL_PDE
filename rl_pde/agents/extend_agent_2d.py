import numpy as np

from rl_pde.policy import Policy

# This could be extended to ND without too much difficulty.
class ExtendAgent2D(Policy):
    """
    Decorator that extends a 1-dimensional agent into 2 dimensions.

    The 1-dimensional agent is applied to every column and every row of a 2-dimensional
    environment.
    """
    def __init__(self, sub_agent):
        self.sub_agent = sub_agent

    def predict(self, state, deterministic=False):
        horizontal_state, vertical_state = state

        h_actions = []
        for index in range(horizontal_state.shape[1]):
            state_slice = horizontal_state[:, index]
            action, _ = self.sub_agent.predict(state_slice, deterministic=deterministic)
            h_actions.append(action)
        horizontal_action = np.stack(h_actions, axis=1)

        v_actions = []
        for index in range(vertical_state.shape[0]):
            state_slice = vertical_state[index, :]
            action, _ = self.sub_agent.predict(state_slice, deterministic=deterministic)
            v_actions.append(action)
        vertical_action = np.stack(v_actions, axis=0)

        return (horizontal_action, vertical_action), None
