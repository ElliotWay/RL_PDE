import gym
import numpy as np

from util.misc import create_stencil_indexes

class SweeperEnv(gym.Env):

    def __init__(self, length=10, max_timesteps=25, accumulate=False, accumulate_max=10):
        self.length = length
        self.accumulate = accumulate
        self.max_timesteps = 25

        self.action_space = gym.spaces.MultiBinary(self.length)
        if self.accumulate:
            self.observation_space = gym.spaces.MultiDiscrete([[accumulate_max] * 3] * self.length)
        else:
            self.observation_space = gym.spaces.MultiBinary((self.length, 3))

        self.reset()

    def _get_observation(self):
        indexes = create_stencil_indexes(stencil_size=3, num_stencils=self.length)
        return self.space[indexes]

    def step(self, action):
        self.timestep += 1

        # Adjacent sweeps cancel.
        action = np.concatenate([[0], action, [0]])
        not_action = np.logical_not(action)
        final_action = np.logical_and(np.logical_and(action[1:-1], not_action[:-2]),
                                        not_action[2:])

        # Sweep to the left.
        # Accumulate mode adds dirt to left adjacent cell.
        if self.accumulate:
            moved_dirt = self.space[1:-1] * final_action

            self.space[1:-1] -= moved_dirt
            self.space[0:-2] += moved_dirt

        # Push mode pushes connected blocks of dirt left.
        else:
            # Not sure how to vectorize this one.
            final_action = np.concatenate([[False], final_action, [False]])
            pushing = False
            for index in reversed(range(len(self.space))):
                if pushing:
                    dirt = self.space[index]
                    if not dirt:
                        self.space[index] = 1
                        pushing = False
                    #else do nothing, this block dirt is being pushed.
                else:
                    sweep = final_action[index]
                    dirt = self.space[index]
                    if sweep and dirt:
                        self.space[index] = 0
                        pushing = True
                    #else do nothing, still searching for a sweep.

        # Take out the trash.
        self.space[0] = 0

        #TODO Add more random dirt?

        state = self._get_observation()
        reward = -self.space[1:-1]
        done = (self.timestep >= self.max_timesteps)
        info = {}
        return state, reward, done, info

    def reset(self):
        #self.space = np.zeros_like(self.space)
        self.space = np.random.randint(2, size=(self.length+2))
        self.space[0] = 0
        self.space[-1] = 0

        self.timestep = 0
        return self._get_observation()

    def render(self):
        print(self.space[1:-1])
        return

    def seed(self):
        return
