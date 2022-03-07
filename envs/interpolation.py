import numpy as np

from util.misc import create_stencil_indexes

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

def weno_interpolation(data, weno_weights, points=1, num_ghosts=None):
    """
    Interpolation based on WENO.

    WENO uses Lagrange polynomials to interpolate flux into the interfaces, then weights between
    them. We can do the same thing with the actual values, then weight the polynomials based on the
    WENO weights.

    Lagrange interpolation is based on the scipy version. We could just use that, but we also need
    a TF compatible version, and we can also vectorize it across points.

    Parameters
    ----------
    data : ndarray with shape [size + num_ghosts]
        Data to interpolate between, including ghost cells.
    weno_weights : ndarray with weights [size+1, 2, WENO order]
        Weights calculated with WENO. This is complicated by having weights for + and - directions.
        Weights are combined by averaging them.
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

    num_points = weno_weights.shape[0] - 1
    order = weno_weights.shape[2]
    if num_ghosts is None:
        num_ghosts = order + 1

    # WENO weights have indices like this:
    # + [-1, 0, 1]
    # -     [0, 1, 2]
    # (Except the - weights are flipped, e.g. [2,1,0].)
    # Averaging these together gives order+1 weights for the order-sized stencils surrounding
    # each interface.
    plus_weights = weno_weights[:, 0]
    minus_weights = np.flip(weno_weights[:, 1], axis=-1)

    # Attach some zeros to facilitate averaging using an offset.
    plus_weights = np.concatenate([plus_weights, np.zeros((order,1))], axis=1)
    minus_weights = np.concatenate([np.zeros((order,1))], minus_weights, axis=1)
    weights = (plus_weights + minus_weights) / 2

    fractional_indexes = [(n+1)/(points+1) for n in range(points)]
    poly_targets = [[index + frac for frac in fractional_indexes] for index in range(order)]
    # poly_targets have shape [interp group, interp point]. Interp group refers to the
    # region between 2 points where we are interpolating, and interp point refers to the specific
    # point we are interpolating to.

    # Polynomial order is one less than the WENO order.
    coef = lagrange_coefficients(order-1, poly_targets)
    # coef has shape [interpolation group, interpolation point, order]

    # num_points + order + 1 stencils because stencils must extend to 'order' points past the real
    # points. One way to think of this is 1 for the stencil all in the left ghosts, and 'order' for
    # each stencil that extends into the right ghosts.
    stencil_indexes = create_stencil_indexes(stencil_size=order,
                                             num_stencils=num_points + order + 1,
                                             offset=num_ghosts - order)
    data_stencils = data[stencil_indexes]
    interpolated = np.sum(data_stencils[:, None, None, :] * coef[None, :, :, :], axis=-1)
    # interpolated has shape [stencil, interp group, interp point].
    # That is, for each stencil, we have a list of (groups of) points we can interpolate with that
    # stencil. We need to reorganize this into a structure where, for each point, we have a list
    # of possible interpolations, which we can then combine with a weighted sum.

    interpolated_per_point = np.array(
            [np.diagonal(interpolated, offset=offset) for offset in range(num_points + 1)])
    weighted_interp = np.sum(interpolated_per_point * weights[:, None, :], axis=-1)

    # Now we just need to insert our interpolated data into the real data.
    almost_real_data = data[num_ghosts - 1:num_ghosts]
    # "almost" because it includes one ghost on the left.
    combined_data = np.concatenate([almost_real_data[:, None], weighted_interp], axis=1).ravel()
    # Then we cut off that ghost.
    combined_data = np.array(combined_data[1:])

    return combined_data
