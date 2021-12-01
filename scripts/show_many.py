import sys
import os
import argparse
import glob
import re
import subprocess
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

def main():
    parser = argparse.ArgumentParser(
        description="Show multiple images in a grid.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("glob", type=str,
            help="Glob string with at most 2 *s. Use quotes at the prompt"
            + " to pass literal *s.")
    parser.add_argument("--output", "-o", type=str, default=None,
            help="Path to save the image grid to. If not specified,"
            + " defaults to showing the grid with eog, then deleting the file.")
    # Could show it with matplotlib instead, I suppose, though that include all the matplotlib
    # interface.

    args = parser.parse_args()
    star_indexes = [i for i, c in enumerate(args.glob) if c == '*']
    num_stars = len(star_indexes)
    if num_stars > 2:
        raise Exception("Too many *s to arrange into a grid.")
    if num_stars == 0:
        raise Exception("Only one file specified - just open the file instead.")

    regex = re.escape(args.glob[:star_indexes[0]])
    regex += "(?P<first>[^*]*)"
    for left_index, right_index in zip(star_indexes[:-1], star_indexes[1:]):
        regex += re.escape(args.glob[left_index+1:right_index])
        regex += "([^*]*)"
    regex += re.escape(args.glob[star_indexes[-1]+1:])
    regex = re.compile(regex)

    files = glob.glob(args.glob)
    nested_dict = {}
    for filename in files:
        m = regex.match(filename)
        parts = regex.match(filename).groups()
        current_dict_level = nested_dict
        for part in parts[:-1]:
            if part in current_dict_level:
                inner_dict = current_dict_level[part]
            else:
                inner_dict = {}
                current_dict_level[part] = inner_dict
            current_dict_level = inner_dict
        assert parts[-1] not in current_dict_level
        current_dict_level[parts[-1]] = filename

    if num_stars == 1:
        sorted_names = sorted(nested_dict.keys())
        fig, axes = plt.subplots(ncols=len(sorted_names))
        for name, ax in zip(sorted_names, axes):
            image = plt.imread(nested_dict[name])
            ax.imshow(image)
            ax.axis('off')
            ax.set_title(name, prop={'size':'small'})
    elif num_stars == 2:
        row_names = sorted(nested_dict.keys())
        col_names_union = set()
        for row in row_names:
            col_names_union |= nested_dict[row].keys()
        col_names = sorted(col_names_union)

        fig, axes = plt.subplots(nrows=len(row_names), ncols=len(col_names),
                figsize=[6.4*len(col_names), 4.8*len(row_names)])
        for row_index, (row_name, ax_row) in enumerate(zip(row_names, axes)):
            subdict = nested_dict[row_name]
            for col_index, (col_name, ax) in enumerate(zip(col_names, ax_row)):
                if col_name not in subdict:
                    continue
                image = plt.imread(subdict[col_name])
                ax.imshow(image)
                ax.axis('off')
                if row_index == 0:
                    ax.text(0.5, 1.0, col_name,
                            transform=ax.transAxes,
                            fontsize='x-large',
                            weight='bold',
                            horizontalalignment='center',
                            verticalalignment='bottom')
                #if row_index == len(row_names) - 1:
                    #ax.set_xlabel(col_name)
                if col_index == 0:
                    ax.text(0.0, 0.5, row_name,
                            transform=ax.transAxes,
                            fontsize='x-large',
                            weight='bold',
                            horizontalalignment='right',
                            verticalalignment='center',
                            rotation='vertical')
                    #ax.set_ylabel(row_name)

    plt.tight_layout()

    if args.output is not None:
        plt.savefig(args.output)
        print(f"Saved plot grid to {args.output}.")
    else:
        temp_name = ".show_many.png"
        plt.savefig(temp_name)
        print(f"Image rendered ({temp_name})...")
        process = subprocess.Popen(["eog", temp_name], 
                stdin=sys.stdin, stdout=sys.stdout, stderr=sys.stderr)
        process.wait()
        os.remove(temp_name)
        print("Done. (Temp file cleared.)")


if __name__ == "__main__":
    main()
