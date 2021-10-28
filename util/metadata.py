import os
import subprocess
import re
import sys
import time
import argparse
from argparse import Namespace

from util.misc import float_dict
from util.git import git_commit_hash, git_is_clean, git_branch_name

META_FILE_NAME = "meta.yaml"
OLD_META_FILE_NAME = "meta.txt"

class MetaFile:
    def __init__(self, log_dir, arg_manager, name=META_FILE_NAME):
        self.path = os.path.join(log_dir, name)
        self.arg_manager = arg_manager
        self.preamble = None
        self.serialized_parameters = None

    def _create_preamble_information(self):
        args = self.arg_manager.args

        preamble_lines = []

        preamble_lines.append("# {}".format(self.path))
        preamble_lines.append("# Experiment Parameter File")

        start_time = time.localtime()
        time_str = time.strftime("%Y/%m/%d, %I:%M:%S %p (%Z), %a", start_time)
        preamble_lines.append("# time started: {}".format(time_str))

        preamble_lines.append("# time finished: ????")
        preamble_lines.append("# status: running")

        try:
            import pwd
            current_user = pwd.getpwuid(os.getuid()).pw_name
            preamble_lines.append("# initiated by user: {}".format(current_user))
        except ImportError:
            preamble_lines.append("# initiated by user: UNKNOWN (run on Windows machine)")

        return_code, commit_id = git_commit_hash()
        if return_code != 0:
            if args.n:
                raise Exception("Git couldn't find HEAD! Are we still in a git repo?")
            elif not args.y:
                _ignore = input("Git couldn't find HEAD."
                                + " Hit <Enter> to continue without recording commit id, or Ctrl-C to stop.")
        else:
            if not re.match("^[0-9a-f]*$", commit_id):
                if not args.y:
                    print("Corrupted commit hash: {}. Fix your git repo before continuing.".format(commit_id))
                    sys.exit(1)
                else:
                    print("Commit hash is corrupted: {}. Something is seriously wrong!".format(commit_id))
            preamble_lines.append("# git commit hash: {}".format(commit_id))

            if git_is_clean():
                preamble_lines.append("# git status: clean")
            else:
                preamble_lines.append("# git status: uncommited changes")

            return_code, branch_name = git_branch_name()
            if return_code != 0:
                if args.n:
                    raise Exception("# Git couldn't find a branch name?")
                elif not args.y:
                    _ignore = input("Git couldn't find a branch name. Hit <Enter> to continue"
                            + " without recording the branch name, or Ctrl-C to stop.")
                preamble_lines.append("# git branch: <error>")
            else:
                preamble_lines.append("# git branch: {}".format(branch_name))

        pid = os.getpid()
        preamble_lines.append("# pid: {}".format(pid))

        self.preamble = "\n".join(preamble_lines)

    def _write(self):
        meta_file = open(self.path, 'w')
        meta_file.write(self.preamble)
        meta_file.write("\n\n")
        meta_file.write(self.serialized_parameters)
        meta_file.write("\n")
        meta_file.close()

    def write_new(self):
        if self.preamble is None:
            self._create_preamble_information()
        self.serialized_parameters = self.arg_manager.serialize()
        self._write()

    def log_finish_time(self, status="finished"):
        end_time = time.localtime()
        time_str = time.strftime("%Y/%m/%d, %I:%M:%S %p (%Z), %a", end_time)

        preamble_lines = self.preamble.split("\n")
        set_finish_time = False
        set_status = False
        for index, line in enumerate(preamble_lines):
            if not set_finish_time and re.search("time finished:", line):
                set_finish_time = True
                new_line = "# time finished: {}".format(time_str)
                preamble_lines[index] = new_line
            elif not set_status and re.search("status:", line):
                set_status = True
                new_line = "# status: {}".format(status)
                preamble_lines[index] = new_line

            if set_finish_time and set_status:
                break

        if not set_finish_time or not set_status:
            print("Meta: Failed to log finish time correctly.")

        self.preamble = "\n".join(preamble_lines)
        self._write()

    def update(self):
        self.serialized_parameters = self.arg_manager.serialize()
        self._write()


# The original function for creating a meta file. This version is left here for reference, as we still
# have a lot of experiments with this format of meta file around.
def create_old_meta_file(log_dir, args):
    #################################################################
    # If adding new metadata, remember that lines MUST end with \n. #
    #################################################################

    meta_filename = os.path.join(log_dir, OLD_META_FILE_NAME)
    meta_file = open(meta_filename, 'x')

    start_time = time.localtime()
    time_str = time.strftime("%Y/%m/%d, %I:%M:%S %p (%Z), %a", start_time)
    meta_file.write("time started: {}\n".format(time_str))

    meta_file.write("time finished: ????\n")
    meta_file.write("status: running\n")

    try:
        import pwd
        current_user = pwd.getpwuid(os.getuid()).pw_name
        meta_file.write("initiated by user: {}\n".format(current_user))
    except ImportError:
        meta_file.write("initiated by user: UNKNOWN (run on Windows machine)\n")

    return_code, commit_id = get_git_commit_id()
    if return_code != 0:
        if args.n:
            raise Exception("Git couldn't find HEAD! Are we still in a git repo?")
        elif not args.y:
            _ignore = input("Git couldn't find HEAD."
                            + " Hit <Enter> to continue without recording commit id, or Ctrl-C to stop.")
    else:
        if not re.match("^[0-9a-f]*$", commit_id):
            if not args.y:
                print("Corrupted commit id: {}. Fix your git repo before continuing.".format(commit_id))
                sys.exit(1)
            else:
                print("Commit id is corrupted: {}. Something is seriously wrong!".format(commit_id))
        meta_file.write("git commit id: {}\n".format(commit_id))

        if is_clean_git_repo():
            meta_file.write("git status: clean\n")
        else:
            meta_file.write("git status: uncommited changes\n")

    pid = os.getpid()
    meta_file.write("pid: {}\n".format(pid))

    arg_dict = vars(args)
    meta_file.write("\nARGUMENTS:\n")
    for k, v in arg_dict.items():
        if k == 'init_params' and v is not None:
            v = ','.join(["{}={}".format(key, value) for key, value in v.items()])
        meta_file.write("{}: {}\n".format(str(k), str(v)))

    meta_file.close()

# The original function for logging the finish time. Kept for reference.
def log_finish_time(log_dir, status="finished"):
    meta_filename = os.path.join(log_dir, OLD_META_FILE_NAME)

    end_time = time.localtime()
    time_str = time.strftime("%Y/%m/%d, %I:%M:%S %p (%Z), %a", end_time)

    meta_file = open(meta_filename, 'r')
    all_lines = [line for line in meta_file]
    meta_file.close()
    no_finish_time = True
    no_status = True
    for index, line in enumerate(all_lines):
        if no_finish_time and re.match("^time finished:", line):
            no_finish_time = False
            new_line = "time finished: {}\n".format(time_str)
            all_lines[index] = new_line
        elif no_status and re.match("^status:", line):
            no_status = False
            new_line = "status: {}\n".format(status)
            all_lines[index] = new_line

        if not (no_finish_time or no_status):
            break

    if no_finish_time:
        all_lines.append("time finished: {}\n".format(time_str))
    if no_status:
        all_lines.append("status: {}\n".format(status))

    meta_file = open(meta_filename, 'w')
    for line in all_lines:
        meta_file.write(line)
    meta_file.close()

# The original function for loading a meta file. Used by load_to_namespace() below.
def load_meta_file(meta_filename):
    meta_file = open(meta_filename)
    meta_dict = {}
    for line in meta_file:
        matches = re.fullmatch("([^:]+):\s*(.+)\n?", line)
        if matches:
            meta_dict[matches.group(1)] = matches.group(2)
    meta_file.close()
    return meta_dict

def destring_value(string):
    try:
        return eval(string)
    except Exception:
        return string

# Not the original function - this is the updated version of the old function to allow backwards
# compatability with existing meta files.
def load_to_namespace(meta_filename, arg_manager, ignore_list=[], override_args=None):
    """
    Load a meta file into an argument manager.

    Arguments in the meta file can be overridden with explicitly specified values. Explicitly
    specified values are loaded from the command line arguments by default, but can be specified
    with override_args. 

    Arguments in the meta file can also be ignored with ignore_list.

    This provides compatability for the new ArgTreeManager system with the original hacky system.
    The original system relies on assumptions I've made about how
    arguments are formatted. Two such assumptions are:
    - bool arguments are all False by default and they are set to true by passing them as flags,
      e.g. --use-thing.
    - the dest argument is never used in ArgumentParser.add_argument(), as this would override the
      default name of the argument in the namespace.

    Parameters
    ----------
    meta_filename : string
        String containing the path to the meta file.
    arg_manager : ArgTreeManager
        Argument manager to load the meta file into.
    ignore_list : iterable of string
        List of parameters to NOT load. These paremeters remain None, even if they had
        no explicit overriding value.
    override_args : dict
        Dictionary of arguments to keep, overriding any value that could be loaded from the meta
        file. By default, use the arguments passed on the command line instead.   

    """

    ALWAYS_IGNORE = ['y', 'n', 'fixed_timesteps', 'memoize']
    ignore_list += ALWAYS_IGNORE

    args = arg_manager.args

    no_default_parser = argparse.ArgumentParser(argument_default=argparse.SUPPRESS)
    def add_to_no_default_parser(args):
        for arg, value in vars(args).items():
            if isinstance(value, Namespace):
                add_to_no_default_parser(value)
            else:
                if arg == 'y' or arg == 'n':
                    continue
                if arg == "fixed_timesteps":
                    continue
                if arg == "output_mode":
                    continue
                arg = "--" + arg
                arg_names = [arg]
                # Allow for arguments with - instead of _.
                if '_' in arg:
                    arg_names.append(arg.replace('_', '-'))
                # Assume bools are flags.
                if isinstance(value, bool):
                    no_default_parser.add_argument(*arg_names, action='store_true')
                else:
                    no_default_parser.add_argument(*arg_names)
    add_to_no_default_parser(args)
    # This is one of the huge problems with this approach.
    no_default_parser.add_argument('--fixed-timesteps', dest='fixed_timesteps', action='store_true',
                        help="Use fixed timesteps. (This is enabled by default.)")
    no_default_parser.add_argument('--variable-timesteps', dest='fixed_timesteps', action='store_false',
                        help="Use variable length timesteps.")
    no_default_parser.add_argument('--memo', dest='memoize', action='store_true', default=None,
                        help="Use a memoized solution to save time. Enabled by default. See --no-memo.")
    no_default_parser.add_argument('--no-memo', dest='memoize', action='store_false', default=None,
                        help="Do not use a memoized solution.")
    if 'output_mode' in args:
        no_default_parser.add_argument('--output-mode', '--output_mode', default=['plot'], nargs='+',
                            help="Type of output from the test. Default 'plot' creates the usual plot"
                            + " files. 'csv' puts the data that would be used for a plot in a csv"
                            + " file. CURRENTLY 'csv' IS NOT IMPLEMENTED FOR ALL OUTPUTS."
                            + " Multiple modes can be used at once, e.g. --output-mode plot csv.")
    no_default_parser.add_argument('-y', '--y', default=False, action='store_true',
                        help="Choose yes for any questions, namely overwriting existing files. Useful for scripts.")
    no_default_parser.add_argument('-n', '--n', default=False, action='store_true',
                        help="Choose no for any questions, namely overwriting existing files."
                        + " Useful for scripts. Overrides the -y option.")

    # Defaults to argv if override_args is None.
    explicit_args, other = no_default_parser.parse_known_args(override_args)
    if len(other) > 0:
        raise Exception("Loading meta file had issues. Couldn't understand these arguments: {}"
                .format(" ".join(other)) + "\n"
                + " Known issues:\n"
                + " * Can't handle explicit arguments with multiple names that differ more\n"
                + " than - vs _.  --num_cells vs --num-cells is fine, but --nx will confuse\n"
                + " it since we don't have a good  way of reconstructing the arg parser except\n"
                + " from the args dict.\n"
                + " * Similarly, parameters with updated names are not loaded correctly,\n"
                + " such as changing --nx to --num-cells. This will assume the defaults,\n"
                + " which is usually the right behavior.\n"
                + " * Boolean parameters that are not default false flags will cause problems.\n"
                + " They need to be handled explicitly."
                + " * Similarly, parameters with nargs>1 will cause problems.\n"
                + " They must also be handled explicitly.")

    meta_dict = load_meta_file(meta_filename)
    if meta_dict['init_params'] != 'None':  # recover dictionary type from strings like 'a=0, b=1'
        meta_dict['init_params'] = float_dict(meta_dict['init_params'])
        # meta_dict['init_params'] = dict(item.split("=") for item in meta_dict['init_params'].split(", "))

    def load_into_nested_namespace(args, meta_dict):
        arg_dict = vars(args)
        sub_namespaces = []
        only_in_self = []
        ignored = []
        explicit = []
        for arg in arg_dict:
            if isinstance(arg_dict[arg], Namespace):
                sub_namespaces.append(arg_dict[arg])
            elif arg not in meta_dict:
                only_in_self.append(arg)
            elif arg in ignore_list:
                ignored.append(arg)
            elif arg in explicit_args:
                explicit.append(arg)
            else:
                arg_dict[arg] = destring_value(meta_dict[arg])

        if len(only_in_self) > 0:
            print("M Some parameters were not found in loaded file: "
                    + ", ".join([f"{name}: {arg_dict[name]}" for name in only_in_self]))
        if len(ignored) > 0:
            print("M Intentionally ignoring some parameters: "
                    + ", ".join(ignored))
        if len(explicit) > 0:
            print("M Explicit parameters overriding loaded parameters: "
                    + ", ".join(explicit))

        for sub_args in sub_namespaces:
            load_into_nested_namespace(sub_args, meta_dict)
    load_into_nested_namespace(args, meta_dict)
