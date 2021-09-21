import argparse
import random
import re
import numpy as np
import tensorflow as tf


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
