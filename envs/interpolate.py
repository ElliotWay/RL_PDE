import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = "2"
os.environ['CUDA_VISIBLE_DEVICES'] = "-1" #Block GPU for now.
import numpy as np
import tensorflow as tf
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

#from tensorflow.python.ops.gen_array_ops import matrix_diag_part_v2
# tf.linalg.diag_part should refer to matrix_diag_part_v2, but it doesn't.
# See github.com/tensorflow/tensorflow/issues/45203.
# But this doesn't work either!!!

from util.misc import create_stencil_indexes

# For default WENO weights.
from envs.weno_solution import lf_flux_split_nd, weno_weights_nd, tf_lf_flux_split, tf_weno_weights

# This computation could be sped up if the memoization is not enough.
# The denominators are the same for any target location with a given order.
_lagrange_dict = {}
def _lagrange_lookup(order, target):
    if order not in _lagrange_dict:
        _lagrange_dict[order] = {}
    o_dict = _lagrange_dict[order]

    if target not in o_dict:
        constants = []
        # Note that we assume point locations are 0 to order.
        for constant_index in range(order+1):
            product = 1.0
            for location in range(order+1):
                if constant_index != location:
                    product *= (target - location)/(constant_index - location)
            constants.append(product)
        o_dict[target] = constants
    return o_dict[target]

def lagrange_coefficients(order, targets):
    """
    Compute coefficients for Lagrange polynomial interpolation.

    These are related to the a matrices in weno_coefficients, but are not the same. I'm not sure
    why, I can't reverse engineer them exactly and the papers are not clear on where they come
    from.

    Lagrange interpolation works on any set of points and can interpolate points at any x, but
    this requires recalculating for any new set of points. We can preload some of this calculation
    with the assumption that points are all evenly spaced from 0 to n-1, and that the locations of
    interpolated points are at fixed locations.

    The resulting coefficients need only be multiplied by the order+1 points to get the
    interpolated value.

    This function memoizes coefficients so that they are only computed once.

    Parameters
    ----------
    order : int
        Order of the polynomial interpolation. One less than the number of points to interpolate
        between.
    targets : ndarray
        Target locations to interpolate to, assuming that existing points are at 0 to n-1.

    Returns
    -------
    coef : ndarray
        List of the coefficients. The resulting ndarray will have the same shape as targets except
        an additional final dimension of length order+1.
    """
    targets = np.array(targets)
    return np.array([_lagrange_lookup(order, target) for target in targets.flat]).reshape(
                                                            targets.shape + (order+1,))

def _default_weno_weights(data, order, num_ghosts):
    """
    Default WENO weights implementation, adapted mainly from WENOBurgersEnv#_prep_state().
    """
    num_points = len(data) - 2*num_ghosts
    # These functions expect a vector dimension.
    data = data[None, :]
    flux_function = lambda q: 0.5 * q ** 2
    flux = flux_function(data)

    fm, fp = lf_flux_split_nd(flux, data)
    fm = fm[0]
    fp = fp[0]
    fp_stencil_indexes = create_stencil_indexes(stencil_size=order * 2 - 1,
                                                num_stencils=num_points + 1,
                                                offset=num_ghosts - order)
    fm_stencil_indexes = np.flip(fp_stencil_indexes, axis=-1) + 1
    fp_stencils = fp[fp_stencil_indexes]
    fm_stencils = fm[fm_stencil_indexes]

    state = np.stack([fp_stencils, fm_stencils], axis=1)
    weights = weno_weights_nd(state, order)
    return weights

def weno_interpolation(data, weno_weights=None, weno_order=None, points=1, num_ghosts=None):
    """
    Interpolation based on WENO.

    WENO uses Lagrange polynomials to interpolate flux into the interfaces, then weights between
    them. We can do the same thing with the actual values, then weight the polynomials based on the
    WENO weights.

    The weights are tricky because they come in + and - directions. Averaging them together kind of
    makes sense when interpolating a single point halfway between (which is why WENO does it this
    way), but less so for points that are closer to one side than another. Interpolating a shock
    creates a plateau halfway between the two flat areas.
    To get around this, this implementation weights the + and - weights based on how far between
    the real points we are. So 0.25 between them gives a 0.75 weight to + and 0.25 weight to -. In
    other words, we're creating a weighted average of weighted averages.

    Parameters
    ----------
    data : ndarray with shape [size + num_ghosts]
        Data to interpolate between, including ghost cells.
    weno_weights : ndarray with weights [size+1, 2, WENO order]
        Weights calculated with WENO. These weights can by ommitted; they will be calculated using
        the standard WENO scheme and Burgers flux.
        The weights are complicated by having + and - directions;
        they are combined by averaging them.
    weno_order : int
        Only used if weno_weights is None. The WENO order to compute default WENO weights.
    points : int
        Number of points to add, evenly spaced, between the original points.
    num_ghosts : int
        Number of ghost cells. Must be at least the WENO order. By default, order+1, because of
        implementation reasons relating to stencils of interfaces (I think).

    Returns
    -------
    interpolated_data : ndarray with shape [points + (points + 1) * size]
        Data with interpolated points. Does NOT include ghost cells, but will include
        interpolated points beyond each boundary if 'points' > 1.
    """

    data = np.array(data)

    if weno_weights is None:
        if num_ghosts is None:
            num_ghosts = weno_order + 1
        weno_weights = _default_weno_weights(data, weno_order, num_ghosts)

    order = weno_weights.shape[2]
    if num_ghosts is None:
        num_ghosts = order + 1
    num_points = len(data) - 2*num_ghosts
    assert (weno_weights.shape[0] == (num_points + 1))

    fractional_indexes = [(n+1)/(points+1) for n in range(points)]
    poly_targets = [[index + frac for frac in fractional_indexes]
                        for index in (np.arange(order+1)-1)]
    # poly_targets have shape [interp group, interp point]. Interp group refers to the
    # region between 2 points where we are interpolating, and interp point refers to the specific
    # point we are interpolating to.

    # Polynomial order is one less than the WENO order.
    coef = lagrange_coefficients(order-1, poly_targets)
    # coef has shape [interpolation group, interpolation point, order]

    # WENO weights have indices like this:
    # + [-1, 0, 1]
    # -     [0, 1, 2]
    # (Except the - weights are flipped, e.g. [2,1,0].)
    # Averaging these together gives order+1 weights for the order-sized stencils surrounding
    # each interface. This average is, in turn, a weighted average based where between points we
    # are interpolating (a simple average for one point halfway).
    plus_weights = weno_weights[:, 0]
    minus_weights = np.flip(weno_weights[:, 1], axis=-1)

    # Attach some zeros to facilitate averaging using an offset.
    plus_weights = np.concatenate([plus_weights, np.zeros((num_points+1,1))], axis=1)
    minus_weights = np.concatenate([np.zeros((num_points+1,1)), minus_weights], axis=1)

    all_weights = np.stack([plus_weights, minus_weights])
    fractional_weights = np.stack([1 - np.array(fractional_indexes), fractional_indexes])
    weights = np.sum(all_weights[:, :, None, :] * fractional_weights[:, None, :, None], axis=0)
    # weights has shape [interface/interp group, stencil, interpolation point]

    # num_stencils=num_points+order+1 because stencils must extend to 'order' points past the real
    # points. One way to think of this is 1 for the stencil all in the left ghosts, and 'order' for
    # each stencil that extends into the right ghosts.
    stencil_indexes = create_stencil_indexes(stencil_size=order,
                                             num_stencils=num_points + order + 1,
                                             offset=num_ghosts - order)
    data_stencils = data[stencil_indexes]
    interpolated = np.sum(data_stencils[:, None, None, :] * coef[None, :, :, :], axis=-1)
    # interpolated has shape [stencil, interp group, interp point].
    # That is, for each stencil, we have a list of (groups of) points we can interpolate with that
    # stencil. We need to reorganize this into a structure where, for each (group of) point(s),
    # we have a list of possible interpolations, which we can then combine with a weighted sum.
    # Simply transposing will not work, however, since each stencil corresponds to a different set
    # of interp groups. To collect all the points for a given interp group, we need to collect
    # along diagonals.
    interpolated = np.flip(interpolated, axis=1) # Flip interp group order so diagonal is correct.
    interpolated_per_point = np.array(
            [np.diagonal(interpolated, offset=-offset) for offset in range(num_points + 1)])
    # interpolated_per_point has shape [interp group, stencil, interp point]
    weighted_interp = np.sum(interpolated_per_point * weights, axis=-1)

    # Now we just need to insert our interpolated data into the real data.
    almost_real_data = data[num_ghosts - 1:-num_ghosts]
    # "almost" because it includes one ghost on the left.
    combined_data = np.concatenate([almost_real_data[:, None], weighted_interp], axis=1).ravel()
    # Then we cut off that ghost.
    combined_data = np.array(combined_data[1:])

    return combined_data

def _tf_default_weno_weights(data, order, num_ghosts):
    # See _default_weno_weights().
    num_points = int(data.shape[0]) - 2*num_ghosts
    # These functions expect a vector dimension.
    data = data[None, :]
    flux = 0.5 * data ** 2

    fm, fp = tf_lf_flux_split(flux, data)
    fm = fm[0]
    fp = fp[0]
    fp_stencil_indexes = create_stencil_indexes(stencil_size=order * 2 - 1,
                                                num_stencils=num_points + 1,
                                                offset=num_ghosts - order)
    fm_stencil_indexes = np.flip(fp_stencil_indexes, axis=-1) + 1
    fp_stencils = tf.gather(fp, fp_stencil_indexes)
    fm_stencils = tf.gather(fm, fm_stencil_indexes)

    state = tf.stack([fp_stencils, fm_stencils], axis=1)
    weights = tf_weno_weights(state, order)
    return weights

def tf_weno_interpolation(data, weno_weights=None, weno_order=None, points=1, num_ghosts=None):
    """
    Interpolation based on WENO. Implemented in TF, see weno_interpolation().

    It is STRONGLY recommended to pass in your own calculated WENO weights instead of using the
    defaults.
    """
    # See weno_interpolation() for additional comments.

    if weno_weights is None:
        if num_ghosts is None:
            num_ghosts = weno_order + 1
        weno_weights = _tf_default_weno_weights(data, weno_order, num_ghosts)

    order = int(weno_weights.shape[2])
    if num_ghosts is None:
        num_ghosts = order + 1
    num_points = int(data.shape[0]) - 2*num_ghosts
    assert (weno_weights.shape[0] == (num_points + 1))

    fractional_indexes = [(n+1)/(points+1) for n in range(points)]
    poly_targets = [[index + frac for frac in fractional_indexes]
                        for index in (np.arange(order+1)-1)]

    # This works because the coefficients are always the same for a given order and number of
    # interpolation points.
    coef = lagrange_coefficients(order-1, poly_targets)

    plus_weights = weno_weights[:, 0]
    minus_weights = tf.reverse(weno_weights[:, 1], axis=[-1])
    plus_weights = tf.concat([plus_weights, np.zeros((num_points+1,1))], axis=1)
    minus_weights = tf.concat([np.zeros((num_points+1,1)), minus_weights], axis=1)
    all_weights = tf.stack([plus_weights, minus_weights])
    # fractional_weights is just a constant, so we can use np.stack.
    fractional_weights = np.stack([1 - np.array(fractional_indexes), fractional_indexes])
    weights = tf.reduce_sum(all_weights[:, :, None, :] * fractional_weights[:, None, :, None], axis=0)

    stencil_indexes = create_stencil_indexes(stencil_size=order,
                                             num_stencils=num_points + order + 1,
                                             offset=num_ghosts - order)
    data_stencils = tf.gather(data, stencil_indexes, axis=-1)
    interpolated = tf.reduce_sum(data_stencils[:, None, None, :] * coef[None, :, :, :], axis=-1)
    interpolated = tf.reverse(interpolated, axis=[1])

    # To my consternation, tf.linalg.diag_part should do what I want,
    # but it is bugged and the k parameter has no effect.
    # matrix_diag_part_v2 should work correctly, and it does during the forward pass, but somehow
    # computing the gradients over this op breaks. I've looked at the source (ops/array_grad.py)
    # for a bug, and if there is one, I can't find it.
    # So we need to implement the diagonal selection operation manually.
    num_stencils, num_interp_groups, _num_points = interpolated.shape
    assert num_stencils == num_points + order + 1
    assert num_interp_groups == order + 1
    shifted_rows = [interpolated[row:num_stencils - num_interp_groups + row + 1, row]
                        for row in range(num_interp_groups)]
    interpolated_per_point = tf.stack(shifted_rows, axis=1)
    # This operation misaligns the axes, need to flip the last 2. I'm still not clear why they end
    # up flipped, but I've run tests and at least the forward pass is working identically to the
    # Numpy version.
    interpolated_per_point = tf.transpose(interpolated_per_point, perm=[0,2,1])
    
    weighted_interp = tf.reduce_sum(interpolated_per_point * weights, axis=-1)

    almost_real_data = data[num_ghosts - 1:-num_ghosts]
    combined_data = tf.concat([almost_real_data[:, None], weighted_interp], axis=1)
    combined_data = tf.reshape(combined_data, shape=[-1])
    combined_data = combined_data[1:]

    return combined_data


