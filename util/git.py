import os
import subprocess

def git_commit_hash():
    try:
        git_proc = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, timeout=1.0)
    except TimeoutError:
        return -1, "timeout"

    output_str = git_proc.stdout.strip()

    return git_proc.returncode, output_str

def git_is_clean():
    """ Returns True if in a clean git repo, False if in a dirty git repo OR an error occurred. """

    return_code = os.system("git diff --quiet")
    return (return_code == 0)

def git_branch_name():
    try:
        git_proc = subprocess.run(["git", "rev-parse", "--abbrev-ref", "HEAD"], capture_output=True,
            text=True, timeout=1.0)
        # Could use git branch --show-current instead on new versions of git.
    except TimeoutError:
        return -1, "timeout"

    output_str = git_proc.stdout.strip()

    return git_proc.returncode, output_str
