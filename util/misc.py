import os
import argparse
import subprocess
import random
import re
import numpy as np
import tensorflow as tf


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

def float_dict(string_dict):
    pairs = string_dict.split(sep=',')
    # empty string returns empty dict
    if len(pairs) <= 1 and len(pairs[0]) == 0:
        return {}
    output_dict = {}
    for pair in pairs:
        match = re.fullmatch("([^=]+)=([^=]+)", pair)
        if not match:
            raise argparse.ArgumentTypeError("In \"{}\", \'{}\' must be key=value.".format(
                string_dict, pair))
        else:
            key = match.group(1)
            value = float(match.group(2))
            output_dict[key] = value
    return output_dict

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

def set_global_seed(seed):
    # I still CANNOT get it to be deterministic.
    # I don't understand what the problem is; it used to give the same results every time,
    # but maybe it only seemed deterministic and was slightly different.
    # Things I've tried to get determinism.
    # * Set these 3 random seeds.
    # * Set the PYTHONHASHSEED at the top of the script.
    # * Restrict tf to 1 CPU thread. (intra and inter_op_parallelism set to 1,
    # set by n_cpu_tf_sess in sac_batch).
    # * Change gate_gradients in minimize functions.
    # Setting the random seeds is still a good idea to get a similar environment,
    # but it's never quite exactly the same.
    np.random.seed(seed)
    tf.set_random_seed(seed)
    random.seed(seed)

def human_readable_time_delta(time_delta, sig_units=0):
    """
    Convert a difference in time from seconds as a floating point to a
    human-readable string, given as a decimal of the most significant unit
    of time. So 4200 will be converted to "1.10hr."

    Useful for when the amount of time has a wide range of magnitudes.
    Inspired by the -h flag for du (the Unix disk usage utility).
    """
    seconds = time_delta
    if seconds < 1:
        millis = seconds * 1000
        return "{:.2f}ms".format(millis)

    minutes = seconds / 60
    if minutes < 1:
        return "{:.2f}s".format(seconds)

    hours = minutes / 60
    if hours < 1:
        return "{:.2f}min".format(minutes)

    days = hours / 24
    if days < 1:
        return "{:.2f}hr".format(hours)

    weeks = days / 7
    if weeks < 1:
        return "{:.2f}d".format(days)

    years = days / 365
    if years < 1:
        return "{:.2f}wk".format(weeks)

    return "{:.2f}yr".format(years)
