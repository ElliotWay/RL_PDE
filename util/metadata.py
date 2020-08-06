import os
import re
import sys
import time

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
    else:
        meta_file.write("initiated by user: UNKNOWN (run on Windows machine)\n")

    # Get commit id of HEAD.
    git_head_file = open('.git/HEAD', 'r')
    current_head = git_head_file.readline()
    git_head_file.close()
    re_match = re.match("^ref: ([\w/]*)$", current_head)
    if re_match is None:
        if args.n:
            raise Exception("Couldn't find git head \".git/HEAD\" to record commit id!".format(args.log_dir))
        elif not args.y:
            _ignore = input("Couldn't find git head \".git/HEAD\" in usual spot."
                            + " Hit <Enter> to continue without recording commit id, or Ctrl-C to stop.")
    else:
        ref_name = re_match.group(1)
        ref_file_name = os.path.join(".git", ref_name)
        ref_file = open(ref_file_name, 'r')
        commit_id = ref_file.readline()
        ref_file.close()
        if not re.match("^[0-9a-f]*$", commit_id):
            print("Corrupted commit id in {}. Fix your git repo before continuing.".format(ref_file_name))
            sys.exit(1)
        else:
            meta_file.write("git commit id: {}".format(commit_id))  # commit_id already has \n

            if os.system("git diff --quiet"):
                meta_file.write("git status: uncommited changes\n")
            else:
                meta_file.write("git status: clean\n")

    pid = os.getpid()
    meta_file.write("pid: {}\n".format(pid))

    arg_dict = vars(args)
    meta_file.write("\nARGUMENTS:\n")
    for k, v in arg_dict.items():
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
