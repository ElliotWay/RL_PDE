import os
import shutil
import sys
import time
import numpy as np

from stable_baselines import logger

from agents import StandardWENOAgent
from util import action_snapshot

def rollout(env, policy, num_rollouts=1, rk4=False, deterministic=False, every_step_hook=None):
    """
    Collect a rollout.

    Parameters
    ----------
    env : Gym environment
        The environment to act in.
    policy : policy with predict function
        Policy to deploy in the environment.
    num_rollouts : int
        Number of rollouts to collect. 1 by default.
    rk4 : bool
        Use RK4 steps instead of regular steps. Requires the environment to have the rk4_step()
        method.
    deterministic : bool
        Require a deterministic policy. Passed to policy.predict().
    every_step_hook : func(t)
        Function to run BEFORE every step, such as rendering the environment. The function should take
        the timestep as an argument.

    Returns
    -------
    (states, actions, rewards, dones, next_states)
    """
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
            if every_step_hook is not None:
                every_step_hook(steps)

            if not rk4:
                action, _ = policy.predict(state, deterministic=deterministic)
                next_state, reward, done, _ = env.step(action)
            else:
                for _ in range(4):
                    action, _ = policy.predict(state)
                    # Only the 4th reward and done are recorded.
                    state, reward, done = env.rk4_step(actions)

            states.append(state)
            actions.append(action)
            rewards.append(reward)
            dones.append(done)
            next_states.append(next_state)

            state = next_state
            steps += 1

    return states, actions, rewards, dones, next_states

def train(env, eval_envs, emi, args):

    action_snapshot.declare_standard_envs(args)

    weno_agent = StandardWENOAgent(order=args.order, mode=args.mode)

    ep_precision = int(np.ceil(np.log(1+args.total_episodes) / np.log(10)))
    log_dir = logger.get_dir()

    training_rewards = []
    training_l2 = []
    total_timesteps = 0
    best_models = []

    #TODO run eval step before any training?

    start_time = time.time()
    for ep in np.arange(args.total_episodes)+1:

        # TODO wrap train step in signal catcher so we can save the model
        # when there is a SIGINT, but not in the middle of training.
        train_info = emi.training_episode(env)

        training_rewards.append(train_info['avg_reward'])
        training_l2.append(env.compute_l2_error())
        total_timesteps += train_info['timesteps']

        if ep % args.log_freq == 0:
            ep_string = ("{:0" + str(ep_precision) + "}").format(ep)

            if args.emi == "batch":
                # The action snapshot doesn't make sense if EMI is not batched,
                # because a standard EMI has a fixed input size.
                action_snapshot.save_action_snapshot(
                        agent=emi.get_policy(), weno_agent=weno_agent,
                        suffix="_ep_" + ep_string)

            # Run eval episodes.
            eval_rewards = []
            eval_l2 = []
            eval_plots = []
            for eval_index, eval_env in enumerate(eval_envs):
                _, _, rewards, _, _ = rollout(eval_env, emi.get_policy(), deterministic=True)
                eval_rewards.append(np.mean(rewards))
                eval_l2.append(eval_env.compute_l2_error())

                eval_suffix = "_ep_" + ep_string
                if len(eval_envs) > 1:
                    eval_suffix = "_eval{}".format(eval_index) + eval_suffix
                plot_file_name = eval_env.plot_state_evolution(
                        num_states=10, full_true=True, suffix=eval_suffix,
                        title="{:03d} training episodes, t = {:05.4f}"
                        .format(ep, eval_env.t))
                eval_plots.append(plot_file_name)

            # Log stats.
            average_train_reward = np.mean(training_rewards)
            average_train_l2 = np.mean(training_l2)
            training_rewards = []
            training_l2 = []
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
            model_file_name = os.path.join(log_dir, "model" + ep_string)
            # probably changes file name by adding e.g. ".zip".
            model_file_name = emi.save_model(model_file_name)
            print("Saved model to " + model_file_name + ".")

            # Keep track of N best models.
            if len(best_models) < args.n_best_models or average_eval_reward > best_models[-1]["reward"]:
                print("New good model ({}).".format(average_eval_reward))
                new_index = -1
                for index, model in enumerate(best_models):
                    if average_eval_reward > model["reward"]:
                        new_index = index
                        break
                best_file_name = os.path.join(log_dir, "_best_model_" + ep_string + ".zip")
                shutil.copy(model_file_name, best_file_name)
                print("Current model copied to {}".format(best_file_name))
                new_model = {"file_name": best_file_name, "episodes": ep,
                        "reward": average_eval_reward, "plots": eval_plots}

                if new_index < 0:
                    best_models.append(new_model)
                else:
                    best_models.insert(new_index, new_model)

                if len(best_models) > args.n_best_models:
                    old_model = best_models.pop()
                    os.remove(old_model["file_name"])
                    print("{} removed.".format(old_model["file_name"]))
        #endif logging
    #endfor episodes

    print("Training complete!")
    print("Training took {:0.1f} seconds, and {} timesteps.".format(
        (time.time() - start_time), total_timesteps))

    # Rename best models based on their final rank.
    print("Best models:")
    for index, model in enumerate(best_models):
        base_best_name = "best_{}_model_{}".format(index+1, model["episodes"])
        new_file_name = os.path.join(log_dir, base_best_name + ".zip")
        shutil.move(model["file_name"], new_file_name)
        print("{}: {} eps, {}, {}".format(index+1, model["episodes"], model["reward"],
            new_file_name))
        for plot_index, plot_file_name in enumerate(model["plots"]):
            new_plot_name = os.path.join(log_dir,
                                base_best_name + "_{}.png".format(plot_index + 1))
            shutil.copy(plot_file_name, new_plot_name)


