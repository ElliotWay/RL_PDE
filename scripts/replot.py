import argparse
import os
import yaml

from rl_pde.run import write_summary_plots, write_final_plots
from util import metadata
from util import param_manager

def main():
    parser = argparse.ArgumentParser(
        description="Regenerate plots based on data collected during training.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("meta", type=str,
            help="Path to the meta file of the experiment for which to replot."
            + " (The experiment data must be in the same directory.)")

    args = parser.parse_args()

    base_meta_name = os.path.basename(args.meta)

    if base_meta_name == metadata.META_FILE_NAME:
        # New file type.
        open_file = open(args.meta, 'r')
        meta_dict = yaml.safe_load(open_file)
        open_file.close()

        total_episodes = meta_dict['total_episodes']
        log_dir = meta_dict['log_dir']
        env_type = meta_dict['env']
        eval_env_type = meta_dict['eval_env']
        init_type = meta_dict['e']['init_type']

    else:
        assert base_meta_name == metadata.OLD_META_FILE_NAME
        # Old file type.
        meta_dict = metadata.load_meta_file(args.meta)
        total_episodes = int(meta_dict['total_episodes'])
        log_dir = meta_dict['log_dir']
        env_type = meta_dict['env']
        eval_env_type = meta_dict['eval_env']
        init_type = meta_dict['init_type']

    summary_plot_dir = os.path.join(log_dir, "summary_plots")
    os.makedirs(summary_plot_dir, exist_ok=True)

    if eval_env_type not in ["std", "custom"]:
        eval_env_names = [init_type]
    else:
        if env_type == "weno_burgers":
            eval_env_names = ['smooth_sine', 'smooth_rare', 'accelshock']
        elif env_type == "weno_burgers_2d":
            eval_env_names = ["gaussian", "1d-smooth_sine-x", "jsz7"]
        else:
            raise Exception("Unknown eval env names.")

    write_summary_plots(log_dir, summary_plot_dir, total_episodes, eval_env_names)
    write_final_plots(log_dir, summary_plot_dir, total_episodes, eval_env_names)


if __name__ == "__main__":
    main()
