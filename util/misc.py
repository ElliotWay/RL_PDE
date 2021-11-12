import sys
import os
import random
import re
import numpy as np
import tensorflow as tf
import subprocess

# I've started using indexes instead of indices because it bothers me when people use "indice" as the singular.
# Between "index and indexes" and "indice and indices" I much prefer the former, so I decided to start using
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

def rescale(values, source, target):
    source_low, source_high = source
    target_low, target_high = target

    descaled = (values - source_low) / (source_high - source_low)

    rescaled = (descaled * (target_high - target_low)) + target_low

    return rescaled

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


class AxisSlice:
    """
    Restrict an ndarray to indexing along a particular axis.

    AxisSlice(a, 1)[index] is (supposed to be) equivalent to a[:, index].
    The advantage of AxisSlice comes when the axis is not known at compile time.

    Examples
    --------
    >>> a = np.arange(12).reshape(3, 4)
    >>> a
    array([[0, 1,  2,  3],
           [4, 5,  6,  7],
           [8, 9, 10, 11]])
    >>> axis0_slice = AxisSlice(a, 0)
    >>> axis0_slice[-2:]
    array([[4, 5, 6, 7],
           [8, 9, 10, 11]])
    >>> axis1_slice = AxisSlice(a, 1)
    >>> axis1_slice[1]
    array([1, 5, 9])
    >>> axis1_slice[(0,3)] = [3,1,4]
    >>> a
    array([[3, 1,  2, 3],
           [1, 5,  6, 1],
           [4, 9, 10, 4]])
    """
    def __init__(self, arr, axis):
        self.arr = arr
        self.axis = axis % arr.ndim # Using "% arr.ndim" handles negative axes.
    def __getitem__(self, indexes):
        return self.arr[(slice(None),) * self.axis + (indexes,)]
    def __setitem__(self, indexes, values):
        # Handle other array-like, but don't make a new copy if already ndarray.
        values = np.array(values, copy=False)

        # If indexes is NOT a single index, but values is shaped to fit a single index,
        # then we need to adjust the shape to broadcast correctly.
        if (not np.issubdtype(type(indexes), np.integer)
                and values.ndim == self.arr.ndim - 1):
            values = np.expand_dims(values, axis=self.axis)

        self.arr[(slice(None),) * self.axis + (indexes,)] = values

def TensorAxisSlice(tensor, axis):
    """
    Like AxisSlice, but for a Tensorflow Tensor instead.

    The Tensor must have a known number of dimensions.
    Note that Tensors cannot use advanced indexing like ndarrays can.

    We could use AxisSlice directly except Tensors don't have a direct 'ndim' property.
    I guess this makes sense since Tensors can have unknown shape. We could use
    len(arr.shape) as an alternative to work for both, but this would be slower, and I prefer if
    AxisSlice is as fast as possible.

    This pretends to be a class, but is actually just a function that adds an 'ndim' property to
    the Tensor and then returns a standard AxisSlice.
    """
    tensor.ndim = tensor.shape.ndims
    return AxisSlice(tensor, axis)

ON_POSIX = 'posix' in sys.builtin_module_names

def soft_link_directories(dir_name, link_name, safe=False):
    """
    Create a symlink between 2 directories.

    If the link already exists, it will be unlinked and deleted before being linked to the new
    directory.
    Works on both Windows and Posix. The requirement that this link only directories, not files, is
    part of a restriction from Windows.

    This is not thread-safe when safe=False. If safe=True, the return code may include errors
    caused by other threads/processes calling this function with the same link_name,
    but it will never raise an exeption.

    Parameters
    ----------
    dir_name : str
        Name of existing directory.
    link_name : str
        Path to link to dir_name.
    safe : bool
        safe=False will raise an Exception if the link cannot be created.
        safe=True will return the error code instead.

    Returns
    -------
    errno : int
        0 if the links were created succesfully, otherwise the error code.
        (Always 0 if safe=False.)
    """
    assert os.path.isdir(dir_name)

    # If the target path is or contains a link, we risk symlink loops by linking directly to it.
    # Instead resolve any links in the target path.
    dir_name = os.path.realpath(dir_name)

    if ON_POSIX:
        try:
            if os.path.islink(link_name):
                os.unlink(link_name)
            os.symlink(dir_name, link_name, target_is_directory=True)
        except OSError as e:
            # One reason you can end up here is if another thread creates the same symlink just
            # after we've unlinked the old one. Generally for this code base, if we're linking to
            # the same file, we don't actually care about the link, so it doesn't matter that we
            # just return the error code and let it go.
            if not safe:
                raise
            else:
                return e.errno
    else:
        # On Windows, creating a symlink requires admin priveleges, but creating
        # a "junction" does not, even though a junction is just a symlink on directories.
        # I think there may be some support in Python3.8 for this,
        # but we need Python3.7 for Tensorflow 1.15.
        try:
            if os.path.isdir(link_name):
                os.rmdir(link_name)
            subprocess.run("mklink /J {} {}".format(link_name, dir_name), shell=True)
        except OSError as e:
            if not safe:
                raise
            else:
                return e.errno

    return 0

