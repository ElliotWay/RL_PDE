import os
import shutil
import sys
import time
import numpy as np

from stable_baselines import logger

from util.action_snapshot import save_action_snapshot

class Model:
    def policy(state, deterministic=False):
        """
        Evaluate the policy on a state to get an action,
        or on a batch of states to get a batch of actions.

        The model should perform any necessary normalization so the action
        can be deployed directly in the environment.
        """
        raise NotImplementedError
    def train(states, actions, rewards, dones, next_states):
        """
        Train using the provided experience.
        The model is not obligated to train; it could store the experience in
        a replay buffer for later.

        The model should perform any necessary normalization
        (or de-normalization) on the samples.
        """
        raise NotImplementedError
    def save(path):
        """
        Save the model to the path.
        """
        raise NotImplementedError


def rollout(env, policy_fn, num_rollouts=1, deterministic=False):
    states = []
    actions = []
    rewards = []
    dones = []
    next_states = []
    for _ in range(num_rollouts):
        state = env.reset()
        done = False
        steps = 0
        while not done:
            action = policy_fn(state)
            next_state, reward, done, _ = env.step(action, deterministic=deterministic)

            states.append(state)
            actions.append(action)
            rewards.append(action)
            dones.append(done)
            next_states.append(next_state)

            state = next_state
            steps += 1

    return states, actions, rewards, dones, next_states

def train(env, eval_envs, model, args):

    weno_agent = StandardWENOAgent(order=args.order, mode=args.mode)

    ep_precision = int(np.ceil(np.log(args.total_episodes) / np.log(10)))
    log_dir = logger.get_dir()

    training_rewards = []
    training_l2 = []
    total_timesteps = 0
    best_models = []

    start_time = time.time()
    for ep in range(args.total_episodes):

        states, actions, rewards, dones, next_states = rollout(env, model.policy)

        # TODO wrap train step in signal catcher so we can save the model
        # when there is a SIGINT, but not in the middle of training.
        model.train(states, actions, rewards, dones, next_states)

        training_rewards.append(np.mean(rewards))
        training_l2.append(env.compute_l2_error())
        total_timesteps += len(states)

        if ep % log_freq == 0:
            ep_string = ("{:0" + str(ep_precision) + "}").format(ep)

            save_action_snapshot(agent=self, weno_agent=weno_agent,
                    suffix="_ep_" + ep_string)

            # Run eval episodes.
            eval_rewards = []
            eval_l2 = []
            eval_plots = []
            for eval_index, eval_env in enumerate(eval_envs):
                _, _, rewards, _, _ = rollout(eval_env, model.policy, deterministic=False)
                eval_rewards.append(np.mean(rewards))
                eval_l2.append(eval_env.compute_l2_error())

                eval_suffix = "_ep_" + ep_string
                if len(eval_envs) > 1:
                    eval_suffix = "_eval{}".format(eval_index) + eval_suffix
                plot_file_name = self.eval_env.plot_state_evolution(
                        num_states=10, full_true=True, suffix=eval_suffix,
                        title="{:03d} training episodes, t = {:05.4f}"
                        .format(ep, eval_env.t))
                eval_plots.append(plot_file_name)

            # Log stats.
            average_train_reward = np.mean(training_rewards)
            average_train_l2 = np.mean(training_l2)
            average_eval_reward = np.mean(eval_rewards)
            average_eval_l2 = np.mean(eval_l2)
            #TODO calculate KL?
            logger.logkv("episodes", ep)
            logger.logkv("avg_train_reward", average_train_reward)
            logger.logkv("avg_train_end_l2", average_train_l2)
            logger.logkv("avg_eval_reward", average_eval_reward)
            logger.logkv("avg_eval_end_l2", average_eval_l2)
            logger.logkv('time_elapsed', int(time.time() - start_time))
            logger.logkv("total_timesteps", total_timesteps)
            logger.dumpkvs()

            # Save model.
            model_file_name = os.path.join(log_dir, "model" + str(num_episodes))
            model.save(model_file_name)
            model_file_name = model_file_name + ".zip"
            print("Saved model to " + model_file_name + ".")

            # Keep track of N best models.
            if len(best_models) < args.n_best_models or average_eval_reward > best_models[-1]["reward"]:
                print("New good model ({}).".format(average_eval_reward))
                new_index = -1
                for index, model in enumerate(best_models):
                    if average_eval_reward > model["reward"]:
                        new_index = index
                        break
                best_file_name = os.path.join(log_dir, "_best_model_" + str(num_episodes) + ".zip")
                shutil.copy(model_file_name, best_file_name)
                new_model = {"file_name": best_file_name, "episodes": ep,
                        "reward": average_eval_reward, "plots": eval_plots}

                if new_index < 0:
                    best_models.append(new_model)
                else:
                    best_models.insert(new_index, new_model)

                if len(best_models) > arg.sn_best_models:
                    old_model = best_models.pop()
                    os.remove(old_model["file_name"])
        #endif log
    #endfor episodes

    print("Training complete!")
    print("Training took {:0.1f} seconds, and {} timesteps.".format(
        (time.time() - start_time), total_timesteps)

    # Rename best models based on their final rank.
    print("Best models:")
    for index, model in enumerate(best_models):
        base_best_name = "best_{}_model_{}".format(index+1, model["episodes"])
        new_file_name = os.path.join(log_dir, base_best_name + ".zip")
        shutil.move(model["file_name"], new_file_name)
        print("{}: {} eps, {}, {}_*".format(index+1, model["episodes"], model["reward"], base_best_name))
        for plot_index, plot_file_name in enumerate(model["plots"]):
            new_plot_name = os.path.join(log_dir,
                                base_best_name + "_{}.png".format(plot_index + 1))
            shutil.copy(plot_file_name, new_plot_name)


