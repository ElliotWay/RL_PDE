import os
import argparse
import yaml
import re

from util import metadata
from util.param_manager import ArgTreeManager
from envs.builder import get_env_arg_parser
from models.builder import get_model_arg_parser

META_IGNORE_KEYS = set(['time started', 'time finished', 'initiated by user', 'pid'])
IGNORE_KEYS = set(['log_dir'])

def main():
    parser = argparse.ArgumentParser(
        description="Print the differences between parameter files. This is cleaner"
        + " than simply using diff. This can also compare old meta.txt files to new meta.yaml"
         + "files, though the results will not be perfect.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("first", type=str,
            help="Path of first meta file to compare.")
    parser.add_argument("second", type=str,
            help="Path of second meta file to compare.")
    args = parser.parse_args()

    def load_file(filename):
        if os.path.basename(filename) == metadata.META_FILE_NAME:
            open_file = open(filename, 'r')
            args_dict = yaml.safe_load(open_file)
            open_file.close()

            arg_manager = ArgTreeManager()
            arg_manager.init_from_dict(args_dict, children_names=['e', 'm'])

            # Load metadata like git hash.
            meta_dict = {}
            open_file = open(filename, 'r')
            for line in open_file:
                meta_param = re.match("# ([^:]+): (.*)", line)
                if meta_param is not None:
                    meta_param_name = meta_param.group(1)
                    meta_param_value = meta_param.group(2)
                    if meta_param_name in ["status", "git status", "git branch"]:
                        meta_dict[meta_param_name] = meta_param_value
                    elif meta_param_name in ["git commit hash", "git commit id"]:
                        meta_dict["git commit hash"] = meta_param_value
            open_file.close()

        elif os.path.basename(filename) == metadata.OLD_META_FILE_NAME:
            env_args = vars(get_env_arg_parser().parse_args(""))
            model_args = vars(get_model_arg_parser().parse_args(""))

            full_dict = metadata.load_meta_file(filename)
            meta_dict = {}
            main_dict = {}
            env_dict = {}
            model_dict = {}
            for k, v in full_dict.items():
                v = metadata.destring_value(v)
                if k in META_IGNORE_KEYS:
                    pass
                elif k in ["status", "git status", "git commit id"]:
                    if k == 'git commit id':
                        k = 'git commit hash'
                    meta_dict[k] = v
                elif k in env_args:
                    env_dict[k] = v
                elif k in model_args:
                    model_dict[k] = v
                else:
                    main_dict[k] = v

            arg_manager = ArgTreeManager()
            arg_manager.init_from_dict(main_dict, children_names=[])

            env_manager = arg_manager.create_child('e')
            env_manager.init_from_dict(env_dict, children_names=[])

            model_manager = arg_manager.create_child('m')
            model_manager.init_from_dict(model_dict, children_names=[])

            # Need to do this manually since we're initializing in an unusual way.
            arg_dict = vars(arg_manager.args)
            arg_dict['e'] = env_manager
            arg_dict['m'] = model_manager
            
        else:
            raise Exception(f"{filename} not a recognized meta file type.")
        return arg_manager, meta_dict

    first_manager, first_meta = load_file(args.first)
    second_manager, second_meta = load_file(args.second)

    if first_meta['status'] == second_meta['status']:
        if first_meta['status'] == "running":
            print("Both files have status: running. Wait until the runs are finished.")
        elif first_meta['status'] != "finished cleanly":
            print(f"Both files have status: {first_meta['status']}, but they may be different in"
                    + " unexpected ways.")
    else:
        print(f"First file has status: {first_meta['status']}."
                + f" Second file has status: {second_meta['status']}.")

    if first_meta['git commit hash'] == second_meta['git commit hash']:
        if first_meta['git status'] != 'clean':
            if second_meta['git status'] != 'clean':
                print("Both files have the same git hash, but both had uncommitted changes,"
                        + " so they may be different in unexpected ways.")
            else:
                print("Both files have the same git hash, but the first had uncommitted changes,"
                        + " so they may be different in unexpected ways.")
        elif second_meta['git status'] != 'clean':
            print("Both files have the same git hash, but the second had uncommitted changes,"
                    + " so they may be different in unexpected ways.")
    else:
        print(f"First file has git hash: {first_meta['git commit hash']}.\n"
                + f" Second file has git hash: {second_meta['git commit hash']}.")
        if first_meta['git status'] == 'clean' and second_meta['git status'] != 'clean':
            print("However, the second file had uncommitted changes; it is possible they were"
                    + " running with the same code anyway.")
        elif first_meta['git status'] != 'clean' and second_meta['git status'] == 'clean':
            print("However, the first file had uncommitted changes; it is possible they were"
                    + " running with the same code anyway.")

    if "git branch" in first_meta:
        if "git branch" in second_meta:
            if first_meta['git branch'] != second_meta['git branch']:
                print(f"First file has git branch: {first_meta['git branch']}."
                        + f" Second file has git branch: {second_meta['git branch']}.")
        else:
            print("First file has the git branch recorded, but the second file does not.")
    elif "git branch" in second_meta:
        print("Second file has the git branch recorded, but the first file does not.")

    def no_diffs(first_manager, second_manager):
        first_arg_dict = vars(first_manager.args)
        second_arg_dict = vars(second_manager.args)
        all_keys = (set(first_arg_dict) | set(second_arg_dict)) - IGNORE_KEYS

        all_children = set(first_manager.children) | set(second_manager.children)
        all_keys = all_keys - all_children
        for key in all_keys:
            if key not in first_arg_dict or key not in second_arg_dict:
                return False
            elif first_arg_dict[key] != second_arg_dict[key]:
                return False
        for child in all_children:
            if not no_diffs(first_manager.get_child(child), second_manager.get_child(child)):
                return False
        return True

    def diff_arg_manager(first_manager, second_manager, prefix=""):
        first_arg_dict = vars(first_manager.args)
        second_arg_dict = vars(second_manager.args)
        all_keys = (set(first_arg_dict) | set(second_arg_dict)) - IGNORE_KEYS

        all_children = set(first_manager.children) | set(second_manager.children)
        all_keys = all_keys - all_children

        for key in all_keys:
            if key in first_arg_dict and key in second_arg_dict:
                first_value = first_arg_dict[key]
                second_value = second_arg_dict[key]
                if first_value != second_value:
                    print(f"{prefix}First file has {key}: {first_value}.\n"
                            + f"{prefix} Second file has {key}: {second_value}.")
            elif key in first_arg_dict:
                print(f"{prefix}{key} missing from second file."
                        + f" First file has {key}: {first_arg_dict[key]}.")
            elif key in second_arg_dict:
                print(f"{prefix}{key} missing from first file."
                        + f" Second file has {key}: {second_arg_dict[key]}.")
            else:
                raise Exception()

        for child in all_children:
            if child in first_manager.children and child in second_manager.children:
                first_child = first_manager.get_child(child)
                second_child = second_manager.get_child(child)
                if not no_diffs(first_child, second_child):
                    if child == 'e':
                        print(f"{prefix}Environment Parameters:")
                    elif child == 'm':
                        print(f"{prefix}Model Parameters:")
                    else:
                        print(f"{prefix}{child} sub-parameters:")
                    diff_arg_manager(first_child, second_child,
                            prefix=(prefix + "  "))
            elif child in first_manager.children:
                print(f"{prefix}Second file missing '{child}' sub-parameters.")
            elif child in second_manager.children:
                print(f"{prefix}First file missing '{child}' sub-parameters.")
            else:
                raise Exception()
    
    diff_arg_manager(first_manager, second_manager)


if __name__ == "__main__":
    main()
