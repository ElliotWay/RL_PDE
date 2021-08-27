import gym
import numpy as np

from util.misc import create_stencil_indexes

class SweeperEnv(gym.Env):

    def __init__(self, length=10, max_timesteps=25):
        self.length = length
        self.max_timesteps = max_timesteps

        self.action_space = gym.spaces.MultiBinary(self.length)
        self.observation_space = gym.spaces.MultiDiscrete([[self.length] * 3] * self.length)
        #self.observation_space = gym.spaces.MultiBinary((self.length, 3))

        self.reset()

    def _get_observation(self, space):
        indexes = create_stencil_indexes(stencil_size=3, num_stencils=self.length)
        return space[indexes]

    def step(self, action):
        self.timestep += 1

        # Adjacent sweeps cancel.
        action = np.concatenate([[0], action, [0]])
        not_action = np.logical_not(action)
        final_action = np.logical_and(np.logical_and(action[1:-1], not_action[:-2]),
                                        not_action[2:])

        # Add dirt to the left adjacent cell.
        moved_dirt = self.space[1:-1] * final_action

        self.space[1:-1] -= moved_dirt
        self.space[0:-2] += moved_dirt

        # Orignal implementation had a "push" mode which pushed connected blocks left;
        # this operation was impossible to vectorize and would result in a extremely large global
        # backprop network.
            #final_action = np.concatenate([[False], final_action, [False]])
            #pushing = False
            #for index in reversed(range(len(self.space))):
                #if pushing:
                    #dirt = self.space[index]
                    #if not dirt:
                        #self.space[index] = 1
                        #pushing = False
                    #else do nothing, this block of dirt is being pushed.
                #else:
                    #sweep = final_action[index]
                    #dirt = self.space[index]
                    #if sweep and dirt:
                        #self.space[index] = 0
                        #pushing = True
                    #else do nothing, still searching for a sweep.

        # Take out the trash.
        self.space[0] = 0

        #TODO Add more random dirt?

        state = self._get_observation(self.space)
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
        return self._get_observation(self.space)

    def render(self):
        print(self.space[1:-1])
        return

    def seed(self):
        return

    def get_real_state(self):
        return self.space.copy()


    @property
    def dimensions(self): return 1

    @property
    def shape(self): return (self.length,)

    @property
    def vec_length(self): return 1

    def tf_prep_state(self, real_state):
        # Use the first (and only) vector component.
        real_state = real_state[0]

        # real_state does not contain the ghost cells on either side.
        state = tf.concat([[0], real_state, [0]], axis=-1)

        rl_state = self._get_observation(state)

        #return rl_state
        return (rl_state,) # Singleton because this is 1D.

    # Could rename this function tf_step; tf_integrate indicates what we're really doing in the
    # other envs.
    def tf_integrate(self, args):
        real_state, rl_state, rl_action = args

        # Only one action part.
        rl_action = rl_action[0]
        # Use the first (and only) vector component.
        real_state = real_state[0]

        # Adjacent sweeps cancel.
        action = tf.concat([[0], action, [0]])
        boolean_action = tf.cast(action, tf.bool)
        not_action = tf.logical_not(boolean_action)
        boolean_final_action = tf.logical_and(
                tf.logical_and(boolean_action[1:-1], not_action[:-2]),
                not_action[2:])

        # Sweep to the left.
        numerical_final_action = tf.cast(boolean_final_action, tf.int32)
        removed_dirt = real_state * numerical_final_action
        added_dirt = tf.concat([removed_dirt[1:], [0]])

        next_real_state = real_state - removed_dirt + added_dirt

        # And random dirt? That would need random input like a VAE.

        # Add back in the vector dimension.
        next_real_state = tf.expand_dims(next_real_state, axis=0)
        return next_real_state

    def tf_calculate_reward(self, args):
        real_state, rl_state, rl_action, next_real_state = args

        reward = next_real_state

        #return reward
        return (reward,) # Singleton because this is 1D.
