import os
import numpy as np

from stable_baselines.sac import SAC, MlpPolicy, LnMlpPolicy
from stable_baselines.common import tf_util, OffPolicyRLModel, SetVerbosity, TensorboardWriter
from stable_baselines.common.math_util import safe_mean, unscale_action, scale_action
from stable_baselines.common.schedules import get_schedule_fn

from models import BaselinesModel
from models.replay_buffer import SB_MARL_ReplayBuffer

class SACModel(BaselinesModel):
    """
    Wrapper for SAC in Stable Baselines.
    
    Initializes SAC, then uses code copied from SAC.learn() to match the functionality of SAC while
    getting samples from an external source.
    It's very hacky, but that's necessary here.

    Things in SAC that are not implemented: callbacks, VecEnv stuff, HER compatability, anything in
    info dict from environment such as successes, some information that SAC records is not recorded
    but as much as possible is passed to caller.
    Things different from SAC: training happens whenever new experience is passed instead of every
    train_freq steps.

    Any bugs in SAC are probably in here too.

    Changing this class is not recommended. If you want a different RL algorithm, write that
    directly, instead of tweaking a wrapper for SAC. That might mean copying SAC and creating
    another wrapper, though.
    """
    def __init__(self, env, args):
        policy_cls = LnMlpPolicy if args.layer_norm else MlpPolicy

        if args.learning_starts is None:
            args.learning_starts = args.ep_length*(args.nx+1)

        policy_kwargs = {'layers':args.layers}
        self.sac = SAC(policy_cls,
                       env,
                       gamma=args.gamma,
                       policy_kwargs=policy_kwargs,
                       learning_rate=args.learning_rate,
                       buffer_size=args.buffer_size,
                       learning_starts=args.learning_starts,
                       batch_size=args.batch_size,
                       train_freq=args.train_freq,
                       verbose=1,
                       #TODO: should this be out of the actual log dir? I don't
                       # actually use tensorboard, so I'm not sure.
                       # It's moot anyway; tb files are taking up space which is silly
                       # given I'm not using them, so I'm disabling these logs.
                       #tensorboard_log="./log/weno_burgers/tensorboard",
                       tensorboard_log=None,
                       )
        self._model = self.sac # Used by superclass.

        # To use the MARL style replay buffer, we need to replace the one SAC is currently using.
        if args.replay_style == "marl":
            print("Using MARL style replay buffer.")
            #TODO: Fix this hack. This class should NOT be aware of the number of agents via this
            # parameter. It should only perceive the individual environment.
            self.sac.replay_buffer = SB_MARL_ReplayBuffer(num_agents=(args.nx+1),
                    size=args.buffer_size)

        sac = self.sac
        sac._setup_learn()
        self.new_tb_log = sac._init_num_timesteps(True)
        # Transform to callable if needed
        sac.learning_rate = get_schedule_fn(sac.learning_rate)

        self.total_timesteps = args.ep_length * args.total_episodes
        self.steps_seen = 0
        self.reward_acc = 0
        self.n_updates = 0
        self.prev_target_update = 0

    def predict(self, obs, deterministic=False):
        # Treat "deterministic" as a training flag.

        sac = self.sac
        assert obs.shape[1:] == sac.observation_space.shape

        vec_obs = sac._is_vectorized_observation(obs, sac.observation_space)
        if not deterministic and (self.steps_seen < sac.learning_starts 
                                    or np.random.rand() < sac.random_exploration):
            # Before training starts, randomly sample actions
            # from a uniform distribution for better exploration.
            # Afterwards, use the learned policy
            # if random_exploration is set to 0 (normal setting)
            if vec_obs:
                unscaled_action = np.array([sac.env.action_space.sample()
                                            for _ in range(len(obs))])
            else:
                unscaled_action = sac.env.action_space.sample()

        else:
            obs = obs.reshape((-1,) + sac.observation_space.shape)
            action = sac.policy_tf.step(obs, deterministic=deterministic)
            action = action.reshape((-1,) + sac.action_space.shape)

            # Add noise to the action (improve exploration,
            # not needed in general)
            if not deterministic and sac.action_noise is not None:
                noise = np.array([sac.action_noise() for _ in range(len(obs))])
                action = np.clip(action + noise, -1, 1)

            # inferred actions need to be transformed to environment action_space before stepping
            unscaled_action = unscale_action(sac.action_space, action)

            if not vec_obs:
                unscaled_action = unscaled_action[0]

        return unscaled_action, None

    def train(self, s, a, r, s2, done):
        assert done.dtype == np.bool, "Done is not bool. Args may be in wrong order."
        sac = self.sac

        total_individual_steps = np.prod(r.shape)
        self.steps_seen += total_individual_steps
        step = self.steps_seen

        with SetVerbosity(sac.verbose), \
             TensorboardWriter(sac.graph, sac.tensorboard_log, "SAC", self.new_tb_log) \
                 as writer:

            sac.replay_buffer.extend(s, a, r, s2, done)

            # Write reward to tensorboard
            if writer is not None:
                ep_reward = np.array(r).reshape(1, -1)
                ep_done = np.array(done).reshape((1, -1))
                reward_acc = np.array([self.reward_acc])
                reward_acc = tf_util.total_episode_reward_logger(
                        reward_acc, ep_reward, ep_done, writer, step)
                self.reward_acc = reward_acc[0]

            mb_infos_vals = []
            # Update policy, critics and target networks
            # Compute current learning_rate
            frac = min(1.0, 1.0 - step / self.total_timesteps)
            current_lr = sac.learning_rate(frac)
            # SAC trains for gradient_steps every train_freq steps.
            # We need to train the equivalent number of times here.
            for grad_step in range(int(sac.gradient_steps * (total_individual_steps / sac.train_freq))):
                # Break if the warmup phase is not over
                # or if there are not enough samples in the replay buffer
                if not sac.replay_buffer.can_sample(sac.batch_size) \
                   or step < sac.learning_starts:
                    break
                self.n_updates += 1
                # Update policy and critics (q functions)
                mb_infos_vals.append(sac._train_step(step, writer, current_lr))

            # Update target network
            if step - self.prev_target_update > sac.target_update_interval:
                self.prev_target_update = step
                sac.sess.run(sac.target_update_op)
            # Log losses and entropy, useful for monitor training
            if len(mb_infos_vals) > 0:
                infos_values = np.mean(mb_infos_vals, axis=0)
            else:
                infos_values = []

        info_dict = {'n_updates':self.n_updates, 'current_lr': current_lr} 
        if len(infos_values) > 0:
            for (name, val) in zip(sac.infos_names, infos_values):
                info_dict[name] = val

        return info_dict
