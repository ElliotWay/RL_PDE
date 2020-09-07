import os
import sys
import subprocess
import threading
from queue import Queue, Empty
import time
import argparse

# Thanks to this SO answer for non-blocking queues.
# https://stackoverflow.com/a/4896288/2860127

# And this SO answer for these colors: 
# https://stackoverflow.com/a/287944/2860127
class colors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

ON_POSIX = 'posix' in sys.builtin_module_names

MAX_PROCS = 4
SLEEP_TIME = 0.25 # seconds

# This isn't a dictionary so we can preserve order.
values_table = []
values_table.append(("--order", [2, 3]))
values_table.append(("--init-type", ["sine", "smooth_sine", "random", "rarefaction", "smooth_rare", "accelshock"]))#, "schedule", "sample"]))
values_table.append(("--seed", [1, 2, 3]))

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
                print("{}: {}".format(index, next_line), end='')
        except Empty:
            # Expected
            pass

    # Check for finished procs.
    for index in list(procs):
        proc = procs[index]
        ret_val = proc.poll()
        if ret_val is not None:
            if ret_val != 0:
                print("Proc {} finished with nonzero return value: {}.".format(index, ret_val))
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

    build_command_list(0, "", "log")

    command_prefix = "python proc_test.py -n "
    command_prefix = command_prefix.split()

    # Dictionaries instead of lists because procs should have the same index
    # when other procs finish and are removed.
    running_procs = {}
    output_queues = {}

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
        print("{}{}: Starting new process ({}/{}):{}".format(colors.OKGREEN,
                new_index, arg_list_index + 1, len(arg_matrix), colors.ENDC))
        print("{}{}: {}{}".format(colors.OKGREEN, new_index, command_string, colors.ENDC))
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
