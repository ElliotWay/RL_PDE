import os
import subprocess
import re
import sys
import time
import argparse

from util.misc import get_git_commit_id, is_clean_git_repo, float_dict

META_FILE_NAME = "meta.txt"


def create_meta_file(log_dir, args):
    #################################################################
    # If adding new metadata, remember that lines MUST end with \n. #
    #################################################################

    meta_filename = os.path.join(log_dir, META_FILE_NAME)
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


def log_finish_time(log_dir, status="finished"):
    meta_filename = os.path.join(log_dir, META_FILE_NAME)

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

def load_meta_file(meta_filename):
    meta_file = open(meta_filename)
    meta_dict = {}
    for line in meta_file:
        matches = re.fullmatch("([^:]+):\s*(.+)\n?", line)
        if matches:
            meta_dict[matches.group(1)] = matches.group(2)
    meta_file.close()
    return meta_dict

# This version didn't quite work as intended.
def override_argv(meta_arg_name="--repeat"):
    new_argv = sys.argv
    new_argv.append("$override_sentinel")
    if meta_arg_name in sys.argv:
        meta_filename = sys.argv[sys.argv.index(meta_arg_name) + 1]
        meta_dict = load_meta_file(meta_filename)
        for arg, value in meta_dict.items():
            if value != "None":
                arg = "--" + arg
                if arg in sys.argv or arg.replace('_', '-') in sys.argv:
                    print("Explicit {} overriding parameter in {}.".format(arg, meta_filename))
                else:
                    if isinstance(value, bool):
                        new_argv.append(arg)
                    elif value[0] == '[' and value[-1] == ']':
                        values = value[1:-1].split(",")
                        new_argv.append(arg)
                        new_argv += values
                    else:
                        new_argv += [arg, value]
    return new_argv[1:] # Remove name of file.

def destring_value(string):
    try:
        return eval(string)
    except Exception:
        return string

def load_to_namespace(meta_filename, args_ns, ignore_list=[], override_args=None):
    """
    Load a meta file into a namespace, i.e. the args created by an ArgumentParser.

    Arguments in the meta file can be overridden with explicitly specified values. Explicitly
    specified values are loaded from the command line arguments by default, but can be specified
    with override_args. 

    Arguments in the meta file can also be ignored with ignore_list.

    This is a bit of a hack right now - it works because of assumptions I've made about how
    arguments are formatted. Two such assumptions are:
    - bool arguments are all False by default and they are set to true by passing them as flags,
      e.g. --use-thing.
    - the dest argument is never used in ArgumentParser.add_argument(), as this would override the
      default name of the argument in the namespace.

    Parameters
    ----------
    meta_filename : string
        String containing the path to the meta file.
    args_ns : Namespace
        Namespace to load the meta file into, i.e. the args created by the ArgumentParser.
    ignore_list : iterable of string
        List of parameters to NOT load. These paremeters remain None, even if they had
        no explicit overriding value.
    override_args : dict
        Dictionary of arguments to keep, overriding any value that could be loaded from the meta
        file. By default, use the arguments passed on the command line instead.   

    """
    # TODO find a way to get around some of this stuff? Might not be possible.

    print("M Loading from meta file: {}".format(meta_filename))
    print("TODO Clean this up so you don't see a lot of irrelevant information.")

    ALWAYS_IGNORE = ['y', 'n']
    ignore_list += ALWAYS_IGNORE

    # Note that this is not a deep copy - changes to the dict will also change the namespace.
    arg_dict = vars(args_ns)

    no_default_parser = argparse.ArgumentParser(argument_default=argparse.SUPPRESS)
    for arg, value in arg_dict.items():
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
                + " which is usually the right behavior.")

    meta_dict = load_meta_file(meta_filename)
    if meta_dict['init_params'] != 'None':  # recover dictionary type from strings like 'a=0, b=1'
        meta_dict['init_params'] = float_dict(meta_dict['init_params'])
        # meta_dict['init_params'] = dict(item.split("=") for item in meta_dict['init_params'].split(", "))

    for arg in arg_dict:
        if arg not in meta_dict:
            print("M {} not found in meta file - using default/explicit value ({}).".format(arg, arg_dict[arg]))
        elif arg in ignore_list:
            print("M Ignoring {} parameter. (Was {}.)".format(arg, meta_dict[arg]))
        elif arg in explicit_args:
            print("M Explicit {} overriding parameter in {}.".format(arg, meta_filename))
        else:
            arg_dict[arg] = destring_value(meta_dict[arg])

        # Note: If a parameter was in the meta file, but not the arg namespace,
        # it was probably a non-parameter field in the meta file.
