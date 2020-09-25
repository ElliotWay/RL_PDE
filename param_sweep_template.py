##################################################################################################
# This is a template for creating a parameter sweep. Make a copy, don't edit this file directly. #
##################################################################################################

import os
import signal
import sys
import shlex
import subprocess
import threading
from queue import Queue, Empty
import time
import argparse

from util.misc import get_git_commit_id, is_clean_git_repo

# Sometimes this enables colors on Windows terminals.
os.system("")

# Thanks to this SO answer for non-blocking queues.
# https://stackoverflow.com/a/4896288/2860127

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

########################################################################
# Declare the base command here.                                       #
# Any parameters that should be the same for every run should go here. #
########################################################################
base_command = "python run_train.py -n"

###########################################
# Declare your available parameters here. #
###########################################
# This isn't a dictionary so we can preserve order. (Though as of 3.6, order is guaranteed.)
values_table = []
values_table.append(("--order", [2, 3]))
values_table.append(("--init-type", ["smooth_sine", "smooth_rare", "accelshock"]))
values_table.append(("--seed", [1, 2, 3]))
# You can declare parameters with dependent parameters.
# This will add "--eval_env custom" to the parameters when we are also using
# "--init-type schedule".
#values_table.append(("--init-type", ["sine", ("schedule", "--eval_env custom")]))

########################################
# Declare the base log directory here. #
########################################
base_log_dir = "log/param_sweep"

#############################################################
# Adjust the maximum number of simultaneous processes here. #
#############################################################
MAX_PROCS = 4
assert MAX_PROCS <= len(colors.SEQUENCE)

SLEEP_TIME = 0.25 # seconds

ON_POSIX = 'posix' in sys.builtin_module_names

arg_matrix = []
def build_command_list(index, arg_list, log_dir):
    if index == len(values_table):
        arg_list += ["--log-dir", log_dir]
        arg_matrix.append(arg_list)
    else:
        keyword, value_list = values_table[index]
        stripped_keyword = keyword.lstrip("-")
        for value in value_list:
            try:
                value, extra = value
            except (TypeError, ValueError):
                extra = None

            new_arg_list = list(arg_list)
            new_arg_list += [keyword, str(value)]
            if extra is not None:
                # shlex takes care of quoted strings with spaces.
                new_arg_list += shlex.split(extra)
            ####################################################################
            # Some arguments, namely log_dir, need more careful manipulation.  #
            # Right now the log_dir creates subdirectories for each parameter. #
            # Make changes in here if you need such changes.                   #
            ####################################################################
            new_log_dir = os.path.join(log_dir, "{}_{}".format(stripped_keyword, value))
            build_command_list(index + 1, new_arg_list, new_log_dir)

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

    # Check for finished procs.
    for index in list(procs):
        proc = procs[index]
        ret_val = proc.poll()
        if ret_val is not None:
            if ret_val != 0:
                print("{}{}: {}Finished with nonzero return value: {}.{}".format(
                    colors.SEQUENCE[index], index, colors.FAIL, ret_val, colors.ENDC))
            del procs[index]
            del queues[index]

def main():
    parser = argparse.ArgumentParser(
        description="Run copies of a command in parallel with varying parameters.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--run', default=False, action='store_true',
                        help="Perform the actual run. The default is to do a dry run which only"
                        " prints the commands instead of running them.")
    args = parser.parse_args()

    if not args.run:
        print("{}Dry run. Look over the commands that will be run, then use --run to actually run the parameter sweep.{}".format(colors.OKBLUE, colors.ENDC))

    build_command_list(0, "", base_log_dir)

    command_prefix = base_command
    command_prefix = command_prefix.split()

    # Dictionaries instead of lists because procs should have the same index
    # when other procs finish and are removed.
    running_procs = {}
    output_queues = {}

    print("Using {} processes:".format(MAX_PROCS), end='')
    for i in range(MAX_PROCS):
        print(" {}{}{}".format(colors.SEQUENCE[i], i, colors.ENDC), end='')
    print()

    return_code, commit_id = get_git_commit_id()
    if return_code != 0:
        print("{}Not in a git repo. Are you sure that's a good idea?{}".format(colors.WARNING, colors.ENDC))
        original_id = None
    else:
        original_id = commit_id
        if not is_clean_git_repo():
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

    try:
        commands_started = 0
        while len(running_procs) > 0 or commands_started < len(arg_matrix):
            while len(running_procs) >= MAX_PROCS or commands_started == len(arg_matrix):
                time.sleep(SLEEP_TIME)
                check_procs(running_procs, output_queues)

            if commands_started < len(arg_matrix):
                # Check that git repo hasn't changed before starting a new proc.
                return_code, current_id = get_git_commit_id()
                dirty_repo = not is_clean_git_repo()
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
                        if len(running_procs) >= 0:
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
                            print(("{}There are no current runs, but some runs are"
                                + " still enqueued. Hit Enter to continue once the"
                                + " git repo has been fixed. (Or ctrl-C twice to stop.){}")
                                .format(colors.WARNING, colors.ENDC))

                        return_code, current_id = get_git_commit_id()
                        dirty_repo = not is_clean_git_repo()

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
                check_procs(running_procs, output_queues)
        except KeyboardInterrupt:
            print(("{}2nd interrupt signal received. Forwarding interrupt to runs."
                + " Sending an interrupt AGAIN will interrupt this process;"
                + " the current runs might not be stopped.{}")
                .format(colors.WARNING, colors.ENDC))
            for index, proc in running_procs.items():
                proc.send_signal(signal.SIGINT)

            while len(running_procs) > 0:
                time.sleep(SLEEP_TIME)
                check_procs(running_procs, output_queues)




if __name__ == "__main__":
    main()
