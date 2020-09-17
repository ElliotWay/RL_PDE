##################################################################################################
# This is a template for creating a parameter sweep. Make a copy, don't edit this file directly. #
##################################################################################################

import os
import sys
import subprocess
import threading
from queue import Queue, Empty
import time
import argparse

# Sometimes this enables colors on Windows terminals.
os.system("")

# Thanks to this SO answer for non-blocking queues.
# https://stackoverflow.com/a/4896288/2860127

# And this SO answer for these colors. 
# https://stackoverflow.com/a/287944/2860127
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



###########################################
# Declare your available parameters here. #
###########################################
# This isn't a dictionary so we can preserve order.
values_table = []
values_table.append(("--order", [2, 3]))
values_table.append(("--init-type", ["smooth_sine", "smooth_rare", "accelshock"]))
values_table.append(("--seed", [1, 2, 3]))

########################################################################
# Declare the base command here.                                       #
# Any parameters that should be the same for every run should go here. #
########################################################################
base_command = "python run_train.py -n"

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
            new_arg_list = list(arg_list)
            new_arg_list += [keyword, str(value)]
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
                    colors.SEQUENCE[index], colors.FAIL, index, ret_val, colors.ENDC))
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

    for arg_list_index, arg_list in enumerate(arg_matrix):
        while len(running_procs) >= MAX_PROCS:
            time.sleep(SLEEP_TIME)
            check_procs(running_procs, output_queues)

        new_index = -1
        for i in range(MAX_PROCS):
            if i not in running_procs:
                new_index = i
                break
        assert new_index >= 0, "No space for new proc?"

        full_command = command_prefix + arg_list
        command_string = " ".join(full_command)
        print("{}{}: {}Starting new process ({}/{}):{}".format(
            colors.SEQUENCE[new_index], new_index,
            colors.OKGREEN, arg_list_index + 1, len(arg_matrix), colors.ENDC))
        print("{}{}: {}{}{}".format(colors.SEQUENCE[new_index], new_index,
            colors.OKGREEN, command_string, colors.ENDC))
        if args.run:
            proc = subprocess.Popen(full_command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                    #text=True, 
                                    bufsize=1, close_fds=ON_POSIX)
            queue = Queue()
            thread = threading.Thread(target=enqueue_output, args=(proc.stdout, queue))
            thread.start()

            running_procs[new_index] = proc
            output_queues[new_index] = queue

    while len(running_procs) > 0:
        time.sleep(SLEEP_TIME)
        check_procs(running_procs, output_queues)



if __name__ == "__main__":
    main()
