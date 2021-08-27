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

from rl_pde.run import rollout
from util.misc import human_readable_time_delta


def write_summary_plots(log_dir, summary_plot_dir, total_episodes):
    #TODO This is a hack. Consider adapting the SB logger class to our own purposes
    # so we can fetch this file name instead of hardcoding it here.
    csv_file = os.path.join(log_dir, "progress.csv")
    csv_df = pd.read_csv(csv_file, comment='#')

    train_color = 'k'
    episodes = csv_df['episodes']

    reward_fig = plt.figure()
    ax = reward_fig.gca()
    train_reward = csv_df['avg_total_reward']
    ax.plot(episodes, train_reward, color=train_color, label="train")

    reward_fig.legend(loc="lower right")
    ax.set_xlim((0, total_episodes))
    ax.set_title("Total Reward per Episode")
    ax.set_xlabel('episodes')
    ax.set_ylabel('reward')
    ax.grid(True)
    ax.set_yscale('symlog')
    #min_scale = 1e-10
    #ax.yaxis.set_minor_locator(SymmetricalLogLocator(base=10, linthresh=min_scale,
            #subs=[0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 2.0, 3.0, 4.0]))
    #ax.set_ymargin(min_scale)

    reward_filename = os.path.join(summary_plot_dir, "rewards.png")
    reward_fig.savefig(reward_filename)
    plt.close(reward_fig)

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

def train(env, emi, args):

    ep_precision = int(np.ceil(np.log(1+args.total_episodes) / np.log(10)))
    log_dir = logger.get_dir()

    summary_plot_dir = os.path.join(log_dir, "summary_plots")
    os.makedirs(summary_plot_dir)

    training_rewards = []
    total_timesteps = 0
    best_models = []

    start_time = time.time()
    for ep in np.arange(args.total_episodes)+1:

        train_info = emi.training_episode(env)

        avg_reward = np.mean([np.mean(reward_part) for reward_part in train_info['reward']])
        training_rewards.append(avg_reward)
        total_timesteps += train_info['timesteps']

        if ep % args.log_freq == 0:

            ep_string = ("{:0" + str(ep_precision) + "}").format(ep)

            # Log stats.
            average_train_reward = np.mean(training_rewards)
            training_rewards = []
            other_stats = dict(train_info)
            del other_stats['reward']
            del other_stats['l2_error']
            del other_stats['timesteps']
            logger.logkv("episodes", ep)
            logger.logkv("avg_total_reward", average_train_reward)
            logger.logkv('time_elapsed', int(time.time() - start_time))
            logger.logkv("total_timesteps", total_timesteps)
            for key, value in other_stats.items():
                logger.logkv(key, value)
            logger.dumpkvs()

            write_summary_plots(log_dir=log_dir, summary_plot_dir=summary_plot_dir,
                    total_episodes=args.total_episodes)

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
            total_episodes=args.total_episodes)

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


