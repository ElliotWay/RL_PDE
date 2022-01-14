import sys
import os
import argparse
import glob
import re
import pandas as pd
import numpy as np

def main():
    parser = argparse.ArgumentParser(
        description="Locate the best combinations in a parameter sweep."
        + " Search for the lowest values of avg_eval_total_reward."
        + " Writes to stdout, use file redirection if you want to save the output.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("directory", type=str,
            help="Directory with sub-directories. Sub-directories eventually contain"
            + " progress.csv files with 'avg_eval_reward'."
            + " If directories named seed_N are encountered, an average is taken across them."
            + " seed_N should be the final sub-directory containing a progress.csv file.")
    parser.add_argument("--script-mode", action='store_true', default=False,
            help="Produce output more suitable to scripts, with full paths"
            + " on separate lines and no additional information.")
    parser.add_argument("--number", "-n", type=int, default=None,
            help="Top N configurations to list. By default, list them all.")

    args = parser.parse_args()

    def find_best(filename):
        csv_df = pd.read_csv(filename, comment='#')
        episodes = csv_df['episodes']
        eval_avgs = csv_df['avg_eval_total_reward']
        return os.path.dirname(filename), max(eval_avgs)

    def recursive_search(search_dir):
        output = []
        seed_files = []
        other_dirs = []
        for dir_entry in os.scandir(search_dir):
            if dir_entry.is_dir():
                if re.fullmatch("seed_\d+", dir_entry.name):
                    progress_file = os.path.join(dir_entry.path, "progress.csv")
                    seed_files.append(progress_file)
                else:
                    other_dirs.append(dir_entry.path)
            elif dir_entry.is_file():
                if dir_entry.name == "progress.csv":
                    output.append(find_best(dir_entry.path))
        for directory in other_dirs:
            output += recursive_search(directory)
        if len(seed_files) > 0:
            seed_values = [find_best(seed_file)[1] for seed_file in seed_files]
            seed_mean = sum(seed_values) / len(seed_values)
            seed_path = os.path.join(search_dir, "seed*")
            output.append((seed_path, seed_mean))
        return output

    best_values = recursive_search(args.directory)
    values_sorted = sorted(best_values, key=lambda tup: tup[1], reverse=True) # Sort descending.

    if args.number is not None:
        best_values = values_sorted[:args.number]
    else:
        best_values = values_sorted

    for path, value in best_values:
        if args.script_mode:
            print(path)
        else:
            shortened_path = path[len(args.directory):]
            print(f"...{shortened_path}: ({value})")

if __name__ == "__main__":
    main()
