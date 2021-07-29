import numpy as np

from rl_pde.run import rollout
from rl_pde.emi.batch import BatchEMI

# Am I still using this anywhere?
# It's still functional code, but I think this was basically superseded by the global backprop idea.
# (I.e. BatchGlobalEMI.)
class HomogenousMARL_EMI(BatchEMI):
    """
    EMI that structures the environment as a MARL problem with homogenous agents.
    It takes samples from the environment, keeps the time and spatial dimension while flattening
    the rest, and passes this to the model.
    
    It is up to the model to interpret the spatial dimension as simultaneous agents.

    The environment space from the presented to the model is the shape of the individual actions,
    i.e. the flattened shape after removing time and spatial dimensions. This requires the model to
    create homogenous agents that are location-agnostic, even though the model receives information
    about location during training.
    """
    def training_episode(self, env):
        self.policy.save_model_samples()
        _, _, reward, done, raw_new_state = rollout(env, self.policy)
    
        # Training requires the state/actions from the perspective of the model, not the
        # perspective of the environment.
        state, action = self.policy.get_model_samples()
        if self.policy.obs_adjust is not None:
            last_state = self.policy.obs_adjust(raw_new_state[-1])
        else:
            last_state = raw_new_state[-1]
        new_state = state[1:] + [last_state]

        # Unlike with BatchEMI, we don't need to reorder the data. We still need to flatten states
        # and actions, however.
        def partial_flatten(arr):
            arr = np.array(arr)
            flattened_shape = np.prod(arr.shape[2:])
            new_shape = arr.shape[:2] + (flattened_shape,)
            arr = arr.reshape(new_shape)
            return arr
        state = partial_flatten(state)
        action = partial_flatten(action)
        reward = np.array(reward)
        unbatched_done = np.repeat(done, len(state[0])).reshape((-1, len(state[0])))
        new_state = partial_flatten(new_state)
        extra_info = self._model.train(state, action, reward, new_state, unbatched_done)

        avg_total_reward = np.mean(np.sum(reward, axis=0))
        l2_error = env.compute_l2_error()
        timesteps = len(state)
        info_dict = {'reward':avg_total_reward, 'l2_error':l2_error, 'timesteps':timesteps}
        info_dict.update(extra_info)
        return info_dict
