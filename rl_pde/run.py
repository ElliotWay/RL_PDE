import os
import shutil
import sys
import time
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import SymmetricalLogLocator

from stable_baselines import logger

# More imports below rollout() to avoid circular dependency.

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
    Each is a list.
    """
    state_list = []
    action_list = []
    reward_list = []
    done_list = []
    next_state_list = []
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
                rk4_substep_state = state
                for _ in range(4):
                    action, _ = policy.predict(rk4_substep_state)
                    # Only the 4th reward and done are recorded.
                    rk4_substep_state, reward, done = env.rk4_step(action)
                next_state = rk4_substep_state

            state_list.append(state)
            action_list.append(action)
            reward_list.append(reward)
            done_list.append(done)
            next_state_list.append(next_state)

            state = next_state
            steps += 1

    return state_list, action_list, reward_list, done_list, next_state_list


from rl_pde.agents import StandardWENOAgent
from rl_pde.emi import OneDimensionalStencil
from envs.plottable_env import Plottable1DEnv, Plottable2DEnv
from util import action_snapshot
from util.misc import human_readable_time_delta


def write_summary_plots(log_dir, summary_plot_dir, total_episodes, eval_env_names):
    #TODO This is a hack. Consider adapting the SB logger class to our own purposes
    # so we can fetch this file name instead of hardcoding it here.
    csv_file = os.path.join(log_dir, "progress.csv")
    csv_df = pd.read_csv(csv_file, comment='#')

    train_color = 'k'
    eval_color = 'tab:orange'
    envs_colors = ['b', 'r', 'g', 'm', 'c', 'y']

    episodes = csv_df['episodes']

    reward_fig = plt.figure()
    ax = reward_fig.gca()
    train_reward = csv_df['avg_train_total_reward']
    ax.plot(episodes, train_reward, color=train_color, label="train")
    avg_eval_reward = csv_df['avg_eval_total_reward']
    ax.plot(episodes, avg_eval_reward, color=eval_color, label="eval avg")
    if len(eval_env_names) > 1:
        for i, name in enumerate(eval_env_names):
            eval_reward = csv_df['eval{}_reward'.format(i+1)]
            ax.plot(episodes, eval_reward,
                    color=envs_colors[i], ls='--', label=name)

    reward_fig.legend(loc="lower right")
    ax.set_xlim((0, total_episodes))
    ax.set_title("Total Reward per Episode")
    ax.set_xlabel('episodes')
    ax.set_ylabel('reward')
    ax.grid(True)
    #linthresh=0.1
    #ax.set_yscale('symlog', linthreshy=linthresh)
    #ax.yaxis.set_minor_locator(SymmetricalLogLocator(base=10, linthresh=linthresh,
                                                        #subs=[1.0, 0.75, 0.5, 0.25]))
    #low, high = ax.get_ylim()
    #if high < 0:
        #ax.set_ylim(top=0.0)
    ax.set_yscale('symlog')
    min_scale = 1e-10
    ax.yaxis.set_minor_locator(SymmetricalLogLocator(base=10, linthresh=min_scale,
            subs=[0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 2.0, 3.0, 4.0]))
    #ax.set_ymargin(min_scale)

    reward_filename = os.path.join(summary_plot_dir, "rewards.png")
    reward_fig.savefig(reward_filename)
    plt.close(reward_fig)

    l2_fig = plt.figure()
    ax = l2_fig.gca()
    train_l2 = csv_df['avg_train_end_l2']
    ax.plot(episodes, train_l2, color=train_color, label="train")
    avg_eval_l2 = csv_df['avg_eval_end_l2']
    ax.plot(episodes, avg_eval_l2, color=eval_color, label="eval avg")
    if len(eval_env_names) > 1:
        for i, name in enumerate(eval_env_names):
            eval_l2 = csv_df['eval{}_end_l2'.format(i+1)]
            ax.plot(episodes, eval_l2,
                    color=envs_colors[i], ls='--', label=name)

    l2_fig.legend(loc="upper right")
    ax.set_xlim((0, total_episodes))
    ax.set_title("L2 Error with WENO at End of Episode")
    ax.set_xlabel('episodes')
    ax.set_ylabel('L2 error')
    ax.grid(True)
    #linthresh=0.001
    #ax.set_yscale('symlog', linthreshy=linthresh)
    #ax.yaxis.set_minor_locator(SymmetricalLogLocator(base=10, linthresh=linthresh,
                                                        #subs=[1.0, 0.75, 0.5, 0.25]))
    #low, high = ax.get_ylim()
    #if low > 0:
        #ax.set_ylim(bottom=0.0)
    ax.set_yscale('log')

    l2_filename = os.path.join(summary_plot_dir, "l2.png")
    l2_fig.savefig(l2_filename)
    plt.close(l2_fig)

    loss_fig = plt.figure()
    ax = loss_fig.gca()
    if 'loss' in csv_df:
        loss = csv_df['loss']
    elif 'policy_loss' in csv_df:
        loss = -csv_df['policy_loss']
    else:
        raise Exception("Can't find loss in progress.csv file.")
    ax.plot(episodes, loss, color='k')
    ax.set_xlim((0, total_episodes))
    ax.set_title("Loss Function")
    ax.set_xlabel('episodes')
    ax.set_ylabel('loss')
    ax.grid(True)
    ax.set_yscale('log')
    #low, high = ax.get_ylim()
    #if low > 0:
        #ax.set_ylim(bottom=0.0)

    loss_filename = os.path.join(summary_plot_dir, "loss.png")
    loss_fig.savefig(loss_filename)
    plt.close(loss_fig)

    print("Summary plots updated in {}.".format(summary_plot_dir))

def write_final_plots(log_dir, summary_plot_dir, total_episodes, eval_env_names):
    """ Create final plots that are different from the summary plots in some way. """ 
    csv_file = os.path.join(log_dir, "progress.csv")
    csv_df = pd.read_csv(csv_file, comment='#')

    train_color = 'k'
    eval_color = 'tab:orange'
    envs_colors = ['b', 'r', 'g', 'm', 'c', 'y']

    episodes = csv_df['episodes']

    l2_fig = plt.figure()
    ax = l2_fig.gca()
    # Don't plot training L2 or average testing L2.
    #train_l2 = csv_df['avg_train_end_l2']
    #ax.plot(episodes, train_l2, color=train_color, label="train")
    #avg_eval_l2 = csv_df['avg_eval_end_l2']
    #ax.plot(episodes, avg_eval_l2, color=eval_color, label="eval avg")
    for i, name in enumerate(eval_env_names):
        eval_l2 = csv_df['eval{}_end_l2'.format(i+1)]
        # Use solid lines instead of dashed lines for this version.
        ax.plot(episodes, eval_l2,
                color=envs_colors[i], ls='-', label=name)

    l2_fig.legend(loc="upper right")
    ax.set_xlim((0, total_episodes))
    ax.set_title("L2 Error with WENO at End of Episode")
    ax.set_xlabel('episodes')
    ax.set_ylabel('L2 error')
    ax.grid(True)
    ax.set_yscale('log')

    l2_filename = os.path.join(summary_plot_dir, "final_l2.png")
    l2_fig.savefig(l2_filename)
    plt.close(l2_fig)
    print("Final plots created in {}.".format(summary_plot_dir))

def train(env, eval_envs, emi, args):

    action_snapshot.declare_standard_envs(args)

    weno_agent = StandardWENOAgent(order=args.order, mode=args.mode)

    ep_precision = int(np.ceil(np.log(1+args.total_episodes) / np.log(10)))
    log_dir = logger.get_dir()

    eval_env_names = [env.grid.init_type for env in eval_envs]

    summary_plot_dir = os.path.join(log_dir, "summary_plots")
    os.makedirs(summary_plot_dir)

    training_rewards = []
    training_l2 = []
    total_timesteps = 0
    best_models = []

    eval_plot_kwargs = {}
    if isinstance(env, Plottable1DEnv):
        eval_plot_kwargs = {'full_true': True, 'num_states': 10}
    elif isinstance(env, Plottable2DEnv):
        eval_plot_kwargs = {'num_frames': 50}

    #TODO run eval step before any training?

    start_time = time.time()
    for ep in np.arange(args.total_episodes)+1:

        # TODO wrap train step in signal catcher so we can save the model
        # when there is a SIGINT, but not in the middle of training.
        train_info = emi.training_episode(env)

        avg_reward = np.mean([np.mean(reward_part) for reward_part in train_info['reward']])
        training_rewards.append(avg_reward)
        training_l2.append(train_info['l2_error'])
        total_timesteps += train_info['timesteps']

        if ep % args.log_freq == 0:

            ep_string = ("{:0" + str(ep_precision) + "}").format(ep)

            #TODO Temporary hack - replace when vector EMI is implemented. -Elliot
            #if isinstance(emi, OneDimensionalStencil):
            if isinstance(emi, OneDimensionalStencil) and not "euler" in args.env:
                # The action snapshot only makes sense when the underlying policy can be applied to
                # 1D environments of arbitrary length, e.g. if it is applied to each 1D stencil in
                # an environment.
                action_snapshot.save_action_snapshot(
                        agent=emi.get_1D_policy(), weno_agent=weno_agent,
                        suffix="_ep_" + ep_string)

            # Run eval episodes.
            eval_rewards = []
            eval_l2 = []
            eval_plots = []
            for eval_index, eval_env in enumerate(eval_envs):
                _, _, rewards, _, _ = rollout(eval_env, emi.get_policy(), deterministic=True)
                reward_parts = list(zip(*rewards))
                avg_total_reward = np.mean([np.mean(np.sum(reward_part, axis=0))
                                                for reward_part in reward_parts])
                eval_rewards.append(avg_total_reward)
                eval_l2.append(eval_env.compute_l2_error())

                eval_suffix = "_ep_" + ep_string
                if len(eval_envs) > 1:
                    eval_suffix = "_eval{}".format(eval_index) + eval_suffix
                plot_file_name = eval_env.plot_state_evolution(
                        suffix=eval_suffix,
                        title="{:03d} training episodes"#, t = {:05.4f}"
                        .format(ep), #, eval_env.t),
                        **eval_plot_kwargs)
                eval_plots.append(plot_file_name)

            # Log stats.
            average_train_reward = np.mean(training_rewards)
            average_train_l2 = np.mean(training_l2)
            training_rewards = []
            training_l2 = []
            average_eval_reward = np.mean(eval_rewards)
            average_eval_l2 = np.mean(eval_l2)
            other_stats = dict(train_info)
            del other_stats['reward']
            del other_stats['l2_error']
            del other_stats['timesteps']
            #TODO calculate KL?
            logger.logkv("episodes", ep)
            logger.logkv("avg_train_total_reward", average_train_reward)
            logger.logkv("avg_train_end_l2", average_train_l2)
            logger.logkv("avg_eval_total_reward", average_eval_reward)
            logger.logkv("avg_eval_end_l2", average_eval_l2)
            if len(eval_envs) > 1:
                for i in range(len(eval_envs)):
                    logger.logkv("eval{}_reward".format(i+1), eval_rewards[i])
                    logger.logkv("eval{}_end_l2".format(i+1), eval_l2[i])
            logger.logkv('time_elapsed', int(time.time() - start_time))
            logger.logkv("total_timesteps", total_timesteps)
            for key, value in other_stats.items():
                logger.logkv(key, value)
            logger.dumpkvs()

            write_summary_plots(log_dir=log_dir, summary_plot_dir=summary_plot_dir,
                    total_episodes=args.total_episodes, eval_env_names=eval_env_names)

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
    write_summary_plots(log_dir=log_dir, summary_plot_dir=summary_plot_dir,
            total_episodes=args.total_episodes, eval_env_names=eval_env_names)
    if 'Burgers' in str(env):  # Disable final plot for Euler env for now
        write_final_plots(log_dir=log_dir, summary_plot_dir=summary_plot_dir,
                total_episodes=args.total_episodes, eval_env_names=eval_env_names)

    print("Training complete!")
    print("Training took {}, and {} timesteps.".format(
        human_readable_time_delta(time.time() - start_time), total_timesteps))

    # Rename best models based on their final rank.
    print("Best models:")
    for index, model in enumerate(best_models):
        base_best_name = "best_{}_model_{}".format(index+1, model["episodes"])
        new_file_name = os.path.join(log_dir, base_best_name + ".zip")
        shutil.move(model["file_name"], new_file_name)
        print("{}: {} eps, {}, {}".format(index+1, model["episodes"], model["reward"],
            new_file_name))
        for plot_index, plot_file_name in enumerate(model["plots"]):
            _, plot_ext = os.path.splitext(plot_file_name)
            new_plot_name = os.path.join(log_dir,
                    "{}_{}{}".format(base_best_name, plot_index+1, plot_ext))
            shutil.copy(plot_file_name, new_plot_name)


