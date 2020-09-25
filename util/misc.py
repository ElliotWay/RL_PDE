import os
import argparse
import subprocess
import numpy as np


# I've started using indexes instead of indices because it bothers me when people use "indice" as the singular.
# Between "index and indexes" and "indices and indices" I much prefer the former, so I decided to start using
# indexes, Latin plurals be damned.
def create_stencil_indexes(stencil_size, num_stencils, offset=0):
    """
    Calculate the indexes for every stencil.

    TODO: figure out a function to do this with multiple dimensions

    Parameters
    ----------
    stencil_size : int
        Size of an individual stencil.
    num_stencils : int
        Number of stencils. In multiple dimensions, this should be a tuple
        of the number of stencils along each dimension.
    offset : int
        Constant offset to add to every index. Equivalently, the value of the
        smallest index in the leftmost stencil.
        Useful to index into an array with ghost cells.
    
    Returns
    -------
    ndarray
        The array of stencil indexes.
    """

    return offset + np.arange(stencil_size)[None, ...] + np.arange(num_stencils)[..., None]

def positive_float(value):
    fvalue = float(value)
    if fvalue <= 0.0:
        raise argparse.ArgumentTypeError("{} is not positive".format(fvalue))
    return fvalue

def nonnegative_float(value):
    fvalue = float(value)
    if fvalue < 0.0:
        raise argparse.ArgumentTypeError("{} is not non-negative".format(fvalue))
    return fvalue

def positive_int(value):
    ivalue = int(value)
    if ivalue <= 0:
        raise argparse.ArgumentTypeError("{} is not positive".format(ivalue))
    return ivalue

def nonnegative_int(value):
    ivalue = int(value)
    if ivalue < 0:
        raise argparse.ArgumentTypeError("{} is not non-negative".format(ivalue))
    return ivalue

def rescale(values, source, target):
    source_low, source_high = source
    target_low, target_high = target

    descaled = (values - source_low) / (source_high - source_low)

    rescaled = (descaled * (target_high - target_low)) + target_low

    return rescaled

def get_git_commit_id():
    try:
        git_head_proc = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, timeout=1.0)
    except TimeoutError:
        return -1, "timeout"

    output_str = git_head_proc.stdout.strip()

    return git_head_proc.returncode, output_str

def is_clean_git_repo():
    """ Returns True if in a clean git repo, False if in a dirty git repo OR an error occurred. """

    return_code = os.system("git diff --quiet")
    return (return_code == 0)
