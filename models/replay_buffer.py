import numpy as np

from stable_baselines.common.buffers import ReplayBuffer

class SB_MARL_ReplayBuffer(ReplayBuffer):
    """
    Replay buffer that samples on a per-timestep basis for a MARL environment, i.e. a sample will
    include the transitions for every agent.

    This version extends the Stable Baselines replay buffer so it can replace that buffer in Stable
    Baselines code. It is implemented so that individual samples are still the inidividual
    transitions, they are just sampled such that every agent's transitions are collected at the
    same time, and sample sizes are checked to make sure they are divisible by the number of
    agents.
    
    Sampling is designed to act the same as the SB ReplayBuffer, but you will need to wrap training
    to handle adding MARL transitions to this buffer.

    VecEnv normalization is not implemented.
    """
    def __init__(self, num_agents:int, size:int):
        self.num_agents = num_agents
        if not size % num_agents == 0:
            raise ValueError("Buffer size must be divisible by number of agents.")
        super().__init__(size)

    def add(self, obs_t, action, reward, obs_tp1, done):
        assert obs_t.shape[0] == self.num_agents
        # These samples are a batch of size num_agents, we can add them into the buffer by using
        # the extend method instead.
        super().extend(obs_t, action, reward, obs_tp1, done)
        assert self._next_idx % self.num_agents == 0

    def extend(self, obs_t, action, reward, obs_tp1, done):
        for data in zip(obs_t, action, reward, obs_tp1, done):
            self.add(*data)

    def sample(self, batch_size: int, env=None):
        assert batch_size % self.num_agents == 0, ("Sample size ({}) must be divisible by the" 
                + " number of agents ({}).").format(batch_size, self.num_agents)
        assert env is None, "VecEnv stuff is not implemented for MARL style buffer."

        real_batch_size = batch_size // self.num_agents
        real_buf_size = len(self) // self.num_agents
        base_indexes = np.random.randint(real_buf_size, size=real_batch_size)
        
        buf_indexes = ((self.num_agents * base_indexes)[..., None]
                        + np.arange(self.num_agents)[None, ...]).flatten()
        return self._encode_sample(buf_indexes)
