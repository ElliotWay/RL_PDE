##################################################################################################
# This is a template for creating a parameter sweep. Make a copy, don't edit this file directly. #
##################################################################################################

#TODO: handle SIGKILL'd child processes. Right now it seems like it thinks they're still running.

import os
import signal
import sys
import shlex
import subprocess
import threading
from queue import Queue, Empty
import time
import argparse
import itertools
from collections import OrderedDict

#TODO: Implement these locally to have a self-contained file?
from util.git import git_commit_hash, git_is_clean
from util.misc import human_readable_time_delta

boolean_flag = object()
flag_true = object()
flag_false = object()

# Thanks to this SO answer for non-blocking queues.
# https://stackoverflow.com/a/4896288/2860127

########################################################################
# Declare the base command here.                                       #
# Any parameters that should be the same for every run should go here. #
########################################################################
base_command = "python run_train.py -n"

##############################################
# Declare your available parameters here.    #
# Note: tuples are not acceptable arguments. #
##############################################
# OrderedDict so order of arguments is preserved.
# (Not actually necessary, dicts preserve initial order since 3.6 anyway.)
values_table = OrderedDict()
values_table["--order"] = [2, 3]
values_table["--init-type"] = ["smooth_sine", "smooth_rare", "accelshock"]
values_table["--seed"] = [1, 2, 3]
# You can declare parameters with dependent parameters.
# This will add "--eval_env custom" to the parameters when we are also using
# "--init-type schedule".
#values_table["--init-type"] = ["sine", ("schedule", "--eval_env custom")]
# Boolean flags like --variable-timesteps can be varied. Use syntax like this:
#values_table["--variable-timesteps"] = boolean_flag
# Unfortunately I haven't worked out the right way to give boolean flags dependent
# parameters.

###########################################################################
# Declare the log directory and other special parameters here.            #
# These parameters are configured based on the special_args dict below.   #
# Instances of <PATH> in the string will be replaced by a path containing #
# sub-directories for each of the arguments in special_args.              #
###########################################################################
experiment_name, _ = os.path.splitext(sys.argv[0]) # Optional, used as a default.
special_params = {}
special_params["--log-dir"] = os.path.join("log/weno_burgers/", experiment_name, "<PATH>")
#special_params["--agent"] = os.path.join("log/weno_burgers", experiment_name, "<PATH>")

##########################################################################
# Declare which of the main parameters are used to configure the special #
# parameters.                                                            #
# 'all' will use all of them in the existing order.                      #
# special_args['main'] can be changed if some of the parameters should   #
# not be in the actual command, and only used to specify the log-dir     #
# or other special params. For example, you may want to select an agent  #
# that trained with a given parameter, but that parameter should not be  #
# specified during testing.                                              #
##########################################################################
special_args = {}
special_args['main'] = 'all'
special_args['--log-dir'] = 'all'
# This creates the log dir with opposite ordering:
#special_args['--log-dir'] = ['--seed', '--init-type', '--order']
#special_args['--agent'] = 'all'

#############################################################
# Adjust the maximum number of simultaneous processes here. #
#############################################################
MAX_PROCS = 3

SLEEP_TIME = 0.25 # seconds

ON_POSIX = 'posix' in sys.builtin_module_names

def clean_name(value):
    if value is flag_true:
        return 'true'
    elif value is flag_false:
        return 'false'
    elif not isinstance(value, str) and hasattr(value, '__iter__'):
        return '_'.join(clean_name(subvalue) for subvalue in value)
    else:
        return ''.join(x if x not in "\0\ \t\n\\/:=.*\"\'<>|?" else '_'
                            for x in str(value))

arg_matrix = [] # Global value, constructed by build_command_list().
def build_command_list(index=0, arg_dict=None, extra_dict=None):
    if arg_dict is None:
        arg_dict = {}
    if extra_dict is None:
        extra_dict = {}

    if index == len(values_table):
        final_special = {}
        for name, value in special_params.items():
            if "<PATH>" in value:
                if special_args[name] == 'all':
                    keys = values_table.keys()
                else:
                    keys = special_args[name]
                clean_names = [clean_name(arg_dict[key]) for key in keys]
                stripped_keys = [key.lstrip("-") for key in keys]
                path = os.path.join(*[f"{key}_{name}"
                            for key, name in zip(stripped_keys, clean_names)])
                final_value = value.replace("<PATH>", path)
            else:
                final_value = value
            final_special[name] = final_value
        if special_args['main'] != 'all':
            arg_dict = {key:value for key, value in arg_dict.items()
                            if key in special_args['main']}
        arg_list = []
        for name in arg_dict:
            # Setting a flag to false means not adding the flag.
            if arg_dict[name] is flag_false:
                pass
            elif arg_dict[name] is flag_true:
                arg_list += [name]
            else:
                arg_list += [name]
                arg_list += arg_dict[name] # arg value is always a list, though often a singleton.
            arg_list += extra_dict[name]
        for name, value in final_special.items():
            arg_list += [name, value] # But special values are always strings.

        arg_matrix.append(arg_list)
    else:
        keyword = list(values_table)[index]
        value_list = values_table[keyword]
        if value_list is boolean_flag:
            false_arg_dict = dict(arg_dict)
            false_arg_dict[keyword] = flag_false
            false_extra_dict = dict(extra_dict)
            false_extra_dict[keyword] = []
            build_command_list(index + 1, false_arg_dict, false_extra_dict)

            true_arg_dict = dict(arg_dict)
            true_arg_dict[keyword] = flag_true
            true_extra_dict = dict(extra_dict)
            true_extra_dict[keyword] = []
            build_command_list(index + 1, true_arg_dict, true_extra_dict)
        else:
            for value in value_list:
                if isinstance(value, tuple):
                    value, extra = value
                else:
                    extra = None
                new_arg_dict = dict(arg_dict)
                # shlex.split is like the usual .split method on string except
                # it does not split on spaces contained inside quotes.
                new_arg_dict[keyword] = shlex.split(str(value))

                new_extra_dict = dict(extra_dict)
                if extra is not None:
                    new_extra_dict[keyword] = shlex.split(extra)
                else:
                    new_extra_dict[keyword] = []
                build_command_list(index + 1, new_arg_dict, new_extra_dict)

# Sometimes this enables colors on Windows terminals.
os.system("")
# Use BrainSlugs83's comment on https://stackoverflow.com/a/16799175/2860127
# to enable these colors on the Windows console:
# In HKCU\Console create a DWORD named VirtualTerminalLevel and set it to 0x1;
# then restart cmd.exe. You can test it with the following powershell:
# "?[1;31mele ?[32mct ?[33mroni ?[35mX ?[36mtar ?[m".Replace('?', [char]27);
class colors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'
    SEQUENCE = [
            '\033[96m', #cyan
            '\033[93m', #yellow
            '\033[95m', #magenta
            '\033[97m', #white
            '\033[92m', #green
            '\033[91m', #red
            '\033[94m', #blue
            '\033[32m', #dark green
            ]
assert MAX_PROCS <= len(colors.SEQUENCE), f"Not enough colors for {MAX_PROCS} processes."


def enqueue_output(out, q):
    for line in iter(out.readline, b''):
        q.put(line.decode('utf-8'))
        #q.put(line)
    out.close()

def check_procs(procs, queues):
    # Print output waiting from each proc.
    for index, queue in queues.items():
        try:
            while True:
                next_line = queue.get_nowait()
                print("{}{}:{} {}".format(colors.SEQUENCE[index], index, colors.ENDC, next_line), end='')
        except Empty:
            # Expected
            pass

    num_errors = 0

    # Check for finished procs. #TODO: also check for dead procs
    for index in list(procs):
        proc = procs[index]
        ret_val = proc.poll()
        if ret_val is not None:
            if ret_val != 0:
                print("{}{}: {}Finished with nonzero return value: {}.{}".format(
                    colors.SEQUENCE[index], index, colors.FAIL, ret_val, colors.ENDC))
                num_errors += 1
            del procs[index]
            del queues[index]
        else:
            #pid = proc.pid
            #if not psutil.pid_exists(pid):
                #print("{}{}: {}Process no longer exists!!!{}".format(
                    #colors.SEQUENCE[index], index, colors.FAIL, colors.ENDC))
                #del procs[index]
                #del queues[index]
            #else:
                #process = psutil.Process(pid)
                #status = process.status()
                #if not status == psutil.STATUS_RUNNING:
                    #print(("{}{}: {}Process is not running! I can't kill it though, what if it's"
                            #+ " just asleep? Status is {}.{}").format(
                                #colors.SEQUENCE[index], index, colors.FAIL, status, colors.ENDC))
            pass

    return num_errors

def main():
    parser = argparse.ArgumentParser(
        description="Run copies of a command in parallel with varying parameters.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--run', default=False, action='store_true',
        help="Perform the actual run. The default is to do a dry run which only"
        + " prints the commands instead of running them.")
    args = parser.parse_args()

    if not args.run:
        print(f"{colors.OKBLUE}Dry run. Look over the commands that will be run,"
        + f" then use --run to actually run the parameter sweep.{colors.ENDC}")

    build_command_list()

    command_prefix = base_command
    command_prefix = shlex.split(command_prefix)

    # Dictionaries instead of lists because procs should have the same index
    # when other procs finish and are removed.
    running_procs = {}
    output_queues = {}

    print("Using {} processes:".format(MAX_PROCS), end='')
    for i in range(MAX_PROCS):
        print(" {}{}{}".format(colors.SEQUENCE[i], i, colors.ENDC), end='')
    print()

    return_code, commit_id = git_commit_hash()
    if return_code != 0:
        print("{}Not in a git repo. Are you sure that's a good idea?{}".format(colors.WARNING, colors.ENDC))
        original_id = None
    else:
        original_id = commit_id
        if not git_is_clean():
            print("{}git repo is not clean: commit before running.{}".format(colors.WARNING, colors.ENDC))
            if args.run:
                return 0

    # TODO write custom interrupt handler. Currently a potential race condition exists if
    # the interrupt happens to come when we're not in that time.sleep.
    signal.signal(signal.SIGINT, signal.default_int_handler)
    try:
        signal.signal(signal.SIGBREAK, signal.default_int_handler)
    except:
        # We're on a Unix machine that doesn't have SIGBREAK, that's fine.
        pass

    start_time = time.time()

    try:
        commands_started = 0
        commands_with_errors = 0
        while len(running_procs) > 0 or commands_started < len(arg_matrix):
            while (len(running_procs) >= MAX_PROCS 
                    or (commands_started == len(arg_matrix) and len(running_procs) > 0)):
                time.sleep(SLEEP_TIME)
                num_errors = check_procs(running_procs, output_queues)
                commands_with_errors += num_errors

            if commands_started < len(arg_matrix):
                # Check that git repo hasn't changed before starting a new proc.
                return_code, current_id = git_commit_hash()
                dirty_repo = not git_is_clean()
                if (original_id is not None and args.run and
                        (return_code != 0 or current_id != original_id or dirty_repo)):
                    while True:
                        if return_code != 0:
                            print("{}A problem has been detected in the git repo!{}"
                                    .format(colors.WARNING, colors.ENDC))
                        elif current_id != original_id:
                            print("{}The git repo has switched to a different commit! (Was {}, now {}){}"
                                    .format(colors.WARNING, original_id, current_id, colors.ENDC))
                        else:
                            assert dirty_repo
                            print("{}Code has been changed in the git repo!{}".format(colors.WARNING, colors.ENDC))
                        print(("{}New runs cannot be started if the code has changed.\n"
                            + "Completed runs are fine. Current runs are probably"
                            + " fine, as CPython precompiles source files.\n"
                            + "There could be an issue if you have a late import.{}")
                            .format(colors.WARNING, colors.ENDC))
                        if len(running_procs) > 0:
                            print(("{}Current runs will continue until finished.\n"
                                + "If the git repo is fixed before the current"
                                + " runs finish, this will be automatically"
                                + " detected.{}")
                                .format(colors.WARNING, colors.ENDC))
                            current_num_procs = len(running_procs)
                            while len(running_procs) == current_num_procs:
                                time.sleep(SLEEP_TIME)
                                check_procs(running_procs, output_queues)
                        else:
                            _ = input(("{}There are no current runs, but some runs are"
                                + " still enqueued. Hit Enter to continue once the"
                                + " git repo has been fixed. (Or ctrl-C twice to stop.){}")
                                .format(colors.WARNING, colors.ENDC))


                        return_code, current_id = git_commit_hash()
                        dirty_repo = not git_is_clean()

                        if return_code == 0 and current_id == original_id and not dirty_repo:
                            print("{}git repo fixed!{}".format(colors.OKGREEN, colors.ENDC))
                            break

                new_index = -1
                for i in range(MAX_PROCS):
                    if i not in running_procs:
                        new_index = i
                        break
                assert new_index >= 0, "No space for new proc?"

                arg_list = arg_matrix[commands_started]
                full_command = command_prefix + arg_list
                #TODO: Some arguments in the full command have spaces.
                #These are being passed correctly to Popen, but will
                #look wrong here.
                #Add quotes to arguments with spaces.
                command_string = " ".join(full_command)
                print("{}{}: {}Starting new process ({}/{}):{}".format(
                    colors.SEQUENCE[new_index], new_index,
                    colors.OKGREEN, commands_started + 1, len(arg_matrix), colors.ENDC))
                print("{}{}: {}{}{}".format(colors.SEQUENCE[new_index], new_index,
                    colors.OKGREEN, command_string, colors.ENDC))
                if args.run:
                    if ON_POSIX:
                        proc = subprocess.Popen(full_command,
                                                stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                                bufsize=1, close_fds=True, start_new_session=True)
                    else:
                        flags = subprocess.DETACHED_PROCESS | subprocess.CREATE_NEW_PROCESS_GROUP
                        proc = subprocess.Popen(full_command,
                                                stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                                bufsize=1, close_fds=False, creation_flags=flags)

                    queue = Queue()
                    thread = threading.Thread(target=enqueue_output, args=(proc.stdout, queue))
                    thread.start()

                    running_procs[new_index] = proc
                    output_queues[new_index] = queue
                commands_started += 1
    except KeyboardInterrupt:
        print(("{}Interrupt signal received. The current runs will run to completion, but no new"
                + " runs will start.\n"
                + "Send the interrupt signal again to stop current runs now.{}")
                .format(colors.WARNING, colors.ENDC))

        try:
            while len(running_procs) > 0:
                time.sleep(SLEEP_TIME)
                num_errors = check_procs(running_procs, output_queues)
                commands_with_errors += num_errors
        except KeyboardInterrupt:
            print(("{}2nd interrupt signal received. Forwarding interrupt to runs."
                + " Sending an interrupt AGAIN will interrupt this process;"
                + " the current runs might not be stopped.{}")
                .format(colors.WARNING, colors.ENDC))
            for index, proc in running_procs.items():
                proc.send_signal(signal.SIGINT)

            STILL_ALIVE_MAX = 10
            still_alive_count = 0

            while len(running_procs) > 0:
                time.sleep(SLEEP_TIME)
                num_errors = check_procs(running_procs, output_queues)
                # They probably WILL have errors, namely interrupt signal received errors.
                commands_with_errors += num_errors

                still_alive_count +=1
                if still_alive_count >= STILL_ALIVE_MAX:
                    print(f"{colors.WARNING}{len(running_procs)} processes are still running."
                            + f" Sending them an interrupt signal again.{colors.ENDC}")
                    for index, proc in running_procs.items():
                        proc.send_signal(signal.SIGINT)
                    still_alive_count = 0

    print("{}Done! {}/{} processes finished in {}.{}".format(
        colors.OKGREEN, commands_started, len(arg_matrix),
        human_readable_time_delta(time.time() - start_time),
        colors.ENDC))
    if commands_started < len(arg_matrix):
        print("{}{}/{} processes were never started.{}".format(
            colors.WARNING, len(arg_matrix) - commands_started, len(arg_matrix),
            colors.ENDC))
    if commands_with_errors > 0:
        print("{}{}/{} processes had nonzero return values.{}".format(
            colors.FAIL, commands_with_errors, len(arg_matrix), colors.ENDC))

    if commands_with_errors == 0 and commands_started == len(arg_matrix):
        sys.exit(0)
    else:
        sys.exit(1)

if __name__ == "__main__":
    main()
