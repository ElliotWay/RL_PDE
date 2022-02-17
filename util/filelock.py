import os
import time
from timeit import default_timer as timer
from contextlib import contextmanager

@contextmanager
def exclusive_open(filename, *args, timeout=10.0, poll_time=0.1, name=None, **kwargs):
    """
    Open a file for exclusive access. Wait until the file is available.

    Creates a lock file in the same directory as the target file. The presence of the lock file
    indicates that the file is locked. If write access to the directory is not available, raise a
    PermissionError.

    Should the lock file be deleted by some external means, a warning will be printed, but no
    exception will be raised.

    Parameters
    ----------
    filename : str
        String path of the file to open for exclusive access.
    timeout : float
        Time in seconds to wait for file access before giving up and raising a FileExistsError,
        or None to keep checking forever.
    poll_time : float
        Time in seconds to wait between checking if the file has become available.
    name : str
        For debugging, a name for this process, which is written into the lock file.
        Defaults to the current PID. 
    *args, **kwargs : ?
        Other parameters such as 'mode' are passed to open().

    Returns
    -------
    A context manager for the open file.
    """
    file_dir, _ = os.path.split(os.path.abspath(filename))
    if not os.access(file_dir, os.W_OK):
        raise PermissionError(f"Cannot lock {filename}, need write access to {file_dir}.")

    if name is None:
        name = str(os.getpid())

    lock_name = f"{filename}.lock"

    if timeout is not None:
        deadline = timer() + timeout

    while True:
        try:
            open_lock_file = open(lock_name, mode='x')
            break
        except FileExistsError:
            if timer() > deadline:
                raise
            time.sleep(poll_time)

    # Wrap things in a try-finally to make sure we can delete the lock file.
    # Notice that even if something happens while the file is in use, we will still catch it with
    # this finally block.
    try:
        open_lock_file.write(f"{name}\n")
        open_lock_file.close()

        with open(filename, *args, **kwargs) as f:
            yield f
    finally:
        try:
            os.remove(lock_name)
        except FileNotFoundError:
            print(f"Lock file \"{lock_name}\" disappeared!"
                + f"Simultaneous access to \"{filename}\" was possible.",
                file=sys.stderr)


def check_lock(filename):
    """
    Check whether a file is currently locked, i.e. whether an associated lock file exists.

    The file's status may have changed before it is opened; use exclusive_open() to open the locked
    file safely.
    
    Parameters
    ----------
    filename : str
        The name of the possibly locked file.

    Returns:
    status : bool
        True if the file is currently locked, or False otherwise.
    """
    lock_name = f"{filename}.lock"
    return os.path.exists(lock_name)


def clear_lock(filename):
    """
    Delete the lock file for a file.

    Obviously, this will interfere with the operation of exclusive_open(), but may be useful in an
    environment where processes are killed without unlocking the file.

    If the lock file does not exist, nothing happens.
    """
    lock_name = f"{filename}.lock"
    os.remove(lock_name)
