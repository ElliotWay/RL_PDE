import argparse
import os
import re
import yaml
import pandas

from rl_pde.run import write_summary_plots, write_final_plots
from util import metadata
from util import param_manager
from util.misc import soft_link_directories

def main():
    parser = argparse.ArgumentParser(
        description="Regenerate plots based on data collected during training."
        + " Useful if you've changed the plot functions in some way since the run.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("meta", type=str,
            help="Path to the meta file of the experiment for which to replot."
            + " (The experiment data must be in the same directory.)")

    args = parser.parse_args()
    log_dir, file_name = os.path.split(args.meta)
    # Could use the log_dir in the meta file, but the files may have moved since then.

    base_meta_name = os.path.basename(args.meta)

    if base_meta_name == metadata.META_FILE_NAME:
        # New file type.
        open_file = open(args.meta, 'r')
        meta_dict = yaml.safe_load(open_file)
        open_file.close()

        total_episodes = meta_dict['total_episodes']
        env_type = meta_dict['env']
        eval_env_type = meta_dict['eval_env']
        init_type = meta_dict['e']['init_type']

    else:
        assert base_meta_name == metadata.OLD_META_FILE_NAME
        # Old file type.
        meta_dict = metadata.load_meta_file(args.meta)
        total_episodes = int(meta_dict['total_episodes'])
        env_type = meta_dict['env']
        eval_env_type = meta_dict['eval_env']
        init_type = meta_dict['init_type']

    summary_plot_dir = os.path.join(log_dir, "summary_plots")
    os.makedirs(summary_plot_dir, exist_ok=True)

    csv_file_name = os.path.join(log_dir, "progress.csv")
    csv_df = pandas.read_csv(csv_file_name, comment='#')
    column_names = list(csv_df)
    eval_env_names = [re.match("eval_(.*)_end_l2", name).group(1) for name in column_names
                            if re.match("eval_.*_end_l2", name)]
    if len(eval_env_names) == 0:
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

    # Create symlink for convenience.
    if len(log_dir) > 0:
        log_link_name = "last"
        error = soft_link_directories(log_dir, log_link_name, safe=True)
        if error:
            print("Note: Failed to create \"last\" symlink.")


if __name__ == "__main__":
    main()
