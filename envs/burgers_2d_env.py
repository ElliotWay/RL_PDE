import numpy as np
import tensorflow as tf
from gym import spaces

from envs.burgers_env import AbstractBurgersEnv
from envs.plottable_env import Plottable2DEnv

class WENOBurgers2DEnv(AbstractBurgersEnv, Plottable2DEnv):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Define spaces.

    def _prep_state(self):

        # Prepare state.

        self.current_state = state
        return state

    def _rk_substep(self, action):

        # Do substep.
        return step
