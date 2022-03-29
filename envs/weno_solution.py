from enum import Enum

import numpy as np
import tensorflow as tf

from envs.solutions import SolutionBase
import envs.weno_coefficients as weno_coefficients
from envs.grid import AbstractGrid, create_grid
from envs.grid1d import Burgers1DGrid
from util.misc import create_stencil_indexes
from util.misc import AxisSlice

# PreciseWENOSolution was intended so that we could train against a higher-resolution version.
# With the realization that the one-step error was needed to be Markovian, using the high-res
# version would require interpolation.
# Now we have a version of this implemented, but it doesn't even use that functionality. There is
# probably a way to get that to work, but it would mean comparing the donwsampled version instead
# of the interpolated version.

def lf_flux_split_nd(flux_array, values_array, grid_type='Burgers', *args):
    """
    Apply Lax-Friedrichs flux splitting along each dimension of the input.

    The maximum velocity (alpha) is computed per 1-dimensional slice.

    The output is structured like:
    [(-,+), (-,+), ...] (- and + directions for each direction.)
    Exception: in 1 dimension, the output will be the single tuple.
    """
    output = []
    abs_values = np.abs(values_array)
    for axis in range(1, flux_array.ndim): # ndim includes the vector dimension on axis 0.
        if grid_type == 'euler':
            eos_gamma = args[0]
            rho = values_array[0]
            v = values_array[1] / rho
            p = (eos_gamma - 1) * (values_array[2, :] - rho * v ** 2 / 2)
            cs = np.sqrt(eos_gamma * p / rho)
            alpha = np.max(np.abs(v) + cs)  # TODO: Euler 2D may require changes
        else:
            alpha = np.max(abs_values, axis=axis, keepdims=True)

        fm = (flux_array - alpha * values_array) / 2
        fp = (flux_array + alpha * values_array) / 2
        output.append((fm, fp))

    if len(output) == 1:
        return output[0]
    else:
        return output

def tf_lf_flux_split(flux_tensor, values_tensor):
    output = []
    abs_values = tf.abs(values_tensor)
    for axis in range(1, flux_tensor.shape.ndims): # ndims includes the vector dimension on axis 0.
        alpha = tf.reduce_max(abs_values, axis=axis, keepdims=True)
        fm = (flux_tensor - alpha * values_tensor) / 2
        fp = (flux_tensor + alpha * values_tensor) / 2
        output.append((fm, fp))

    if len(output) == 1:
        return output[0]
    else:
        return output


#TODO: Finish this function. It is not currently in use. I couldn't find a clean way of handling
# this in a general case. - direction stencils need to be flipped and shifted. Unsplit flux has a
# different stencil size. Could use same parameters as create_stencil_indexes?
# It would be better if every stenciling used the same function; I've already had bugs from one
# version doing it incorrectly.
def create_stencils_nd(values_array, order, axis, source_grid):
    """
    UNFINISHED DO NOT USE
    Create 1D stencils from an ndarray.

    Ghost cells from axes besides the stenciled axis will be trimmed.

    The source_grid parameter will not be modified - it is used to access the underlying properties.
    """

    # Trim ghost cells of other dimensions.
    ghost_slice = list(source_grid.real_slice)
    ghost_slice[axis] = slice(None)
    trimmed = values_array[ghost_slice]

    indexes = create_stencil_indexes(stencil_size=order * 2 - 1,
                                     num_stencils=source_grid.num_cells[axis] + 1,
                                     offset=source_grid.num_ghosts[axis] - order)

    # How to handle this?
    #left_stencil_indexes = np.flip(right_stencil_indexes, axis=-1) + 1

    # Indexing is tricky here. I couldn't find a way without transposing and then transposing
    # back. (It might be possible, though.)
    left_stencils = (flux_left.transpose()[:, left_stencil_indexes]).transpose([1,0,2])
    right_stencils = (flux_right.transpose()[:, right_stencil_indexes]).transpose([1,0,2])


def weno_sub_stencils_nd(stencils_array, order):
    """
    Interpolate sub-stencils in an ndarray of stencils. (The stencils are 1D.)

    An ndarray of stencils:
    [spatial dimensions... X stencil size]
    ->
    An ndarray of interpolated sub-stencils:
    [spatial dimensions... X num sub-stencils]
    """
    # These weights have shape order X order (i.e. num stencils * stencil size).
    a_mat = weno_coefficients.a_all[order]

    # These weights are "backwards" in the original formulation.
    # This is easier in the original formulation because we can add the k for our kth stencil to the index,
    # then subtract by a variable amount to get each value, but there's no need to do that here, and flipping
    # it back around makes the expression simpler.
    a_mat = np.flip(a_mat, axis=-1)

    sub_stencil_indexes = create_stencil_indexes(stencil_size=order, num_stencils=order)
    sub_stencils = stencils_array[..., sub_stencil_indexes]

    interpolated = np.sum(a_mat * sub_stencils, axis=-1)
    return interpolated

@tf.function
def tf_weno_sub_stencils(stencils_tensor, order):
    a_mat = weno_coefficients.a_all[order]
    a_mat = np.flip(a_mat, axis=-1)

    sub_stencil_indexes = create_stencil_indexes(stencil_size=order, num_stencils=order)
    sub_stencils = tf.gather(stencils_tensor, sub_stencil_indexes, axis=-1)

    interpolated = tf.reduce_sum(a_mat * sub_stencils, axis=-1)
    return interpolated


# Future note: we're wasting some computations.
# The squared matrix for the kth sub-stencil has a lot of overlap with the q^2 matrix with the k+1st sub-stencil.
# Also the upper triangle of the q^2 matrix is not needed.
# Not a huge deal, but could be relevant with higher orders or factors of increasing complexity.
def weno_weights_nd(stencils_array, order):
    """
    Compute standard WENO weights for a given ndarray of stencils. (The stencils are 1D.)

    An ndarray of stencils:
    [spatial dimensions... X stencil size]
    An ndarray of weights:
    [spatial dimensions X num weights (one for each sub-stencil)
    """

    C = weno_coefficients.C_all[order]
    sigma = weno_coefficients.sigma_all[order]
    epsilon = 1e-16

    sub_stencil_indexes = create_stencil_indexes(stencil_size=order, num_stencils=order)
    sub_stencils = AxisSlice(stencils_array, -1)[sub_stencil_indexes]

    # "beta" below refers to the smoothness indicator.
    # To my best understanding, beta is calculated based on an approximation of the square of pairs
    # of values in the stencil, so we multiply every pair of values in a sub-stencil together,
    # weighted appropriately for the approximation by sigma.

    # Flip the stencils because in the original formulation we subtract from the last index instead
    # of adding from the first, and sigma weights assume this ordering.
    sub_stencils_flipped = np.flip(sub_stencils, axis=-1)

    # Insert extra empty dimensions so numpy broadcasting produces
    # the desired outer product along only the intended dimensions.
    squared = sub_stencils_flipped[..., None, :] * sub_stencils_flipped[..., :, None]
    #print("max flux:", np.max(squared))
    # Note that sigma is made up of triangular matrices, so the second copy of each pair is weighted by 0.
    beta = np.sum(sigma * squared, axis=(-2, -1))
    #print("max beta:", np.max(beta))
    #if np.max(beta) == 0.0:
        #print(sub_stencils)
        #assert False

    alpha = C / (epsilon + beta ** 2)
    # This uses a slightly different version. Why? Is computing WENO for Euler somehow different?
    # https://github.com/python-hydro/hydro_examples/blob/master/compressible/weno_euler.py#L97
    # alpha = C / (epsilon + np.abs(beta) ** order)

    # Keep the sub-stencil dimension after the sum so numpy broadcasts there
    # instead of on the stencil dimension.
    weights = alpha / np.sum(alpha, axis=-1, keepdims=True)

    return weights

@tf.function
def tf_weno_weights(stencils_tensor, order):
    C = weno_coefficients.C_all[order]
    sigma = weno_coefficients.sigma_all[order]
    epsilon = 1e-16

    sub_stencil_indexes = create_stencil_indexes(stencil_size=order, num_stencils=order)
    sub_stencils = tf.gather(stencils_tensor, sub_stencil_indexes, axis=-1)
    # For some reason, axis for tf.reverse MUST be iterable, can't be scalar -1.
    sub_stencils_flipped = tf.reverse(sub_stencils, axis=(-1,))

    squared = (tf.expand_dims(sub_stencils_flipped, axis=-2)
                * tf.expand_dims(sub_stencils_flipped, axis=-1))
    beta = tf.reduce_sum(sigma * squared, axis=(-2, -1))
    alpha = C / (epsilon + beta ** 2)
    # alpha = C / (epsilon + tf.abs(beta) ** order)
    weights = alpha / tf.reduce_sum(alpha, axis=-1, keepdims=True)
    return weights


def weno_reconstruct_nd(order, stencils_array):
    sub_stencils = weno_sub_stencils_nd(stencils_array, order)
    weights = weno_weights_nd(stencils_array, order)

    reconstructed = np.sum(weights * sub_stencils, axis=-1)
    return reconstructed, weights

@tf.function
def tf_weno_reconstruct(stencils_tensor, order):
    sub_stencils = tf_weno_sub_stencils(stencils_tensor, order)
    weights = tf_weno_weights(stencils_tensor, order)

    reconstructed = tf.reduce_sum(weights * sub_stencils, axis=-1)
    return reconstructed, weights

def revecs_vectorized(g, boundary_state):
    real_length = g.nx + 1
    revecs = np.zeros((3, 3, real_length))
    rho = boundary_state[0, :]  # np.zeros((real_length))
    S = boundary_state[1, :]  # np.zeros((real_length))
    E = boundary_state[2, :]  # np.zeros((real_length))
    v = S / rho
    p = (g.eos_gamma - 1) * (E - rho * v ** 2 / 2)
    cs = np.sqrt(g.eos_gamma * p / rho)
    revecs[0, 0, :] = 1
    revecs[0, 1, :] = v - cs
    revecs[0, 2, :] = (E + p) / rho - v * cs
    revecs[1, 0, :] = 1
    revecs[1, 1, :] = v
    revecs[1, 2, :] = v ** 2 / 2
    revecs[2, 0, :] = 1
    revecs[2, 1, :] = v + cs
    revecs[2, 2, :] = (E + p) / rho + v * cs
    return revecs

def levecs_vectorized(g, boundary_state):
    real_length = g.nx + 1
    levecs = np.zeros((3, 3, real_length))
    rho = boundary_state[0, :]  # np.zeros((real_length))
    S = boundary_state[1, :]  # np.zeros((real_length))
    E = boundary_state[2, :]  # np.zeros((real_length))
    v = S / rho
    p = (g.eos_gamma - 1) * (E - rho * v ** 2 / 2)
    cs = np.sqrt(g.eos_gamma * p / rho)
    b1 = (g.eos_gamma - 1) / cs ** 2
    b2 = b1 * v ** 2 / 2
    levecs[0, 0, :] = (b2 + v / cs) / 2
    levecs[0, 1, :] = -(b1 * v + 1 / cs) / 2
    levecs[0, 2, :] = b1 / 2
    levecs[1, 0, :] = 1 - b2
    levecs[1, 1, :] = b1 * v
    levecs[1, 2, :] = -b1
    levecs[2, 0, :] = (b2 - v / cs) / 2
    levecs[2, 1, :] = -(b1 * v - 1 / cs) / 2
    levecs[2, 2, :] = b1 / 2
    return levecs

class RKMethod(Enum):
    EULER = 1
    RK4 = 2
    RK3 = 3
    SSP_RK3 = 3
    RK8 = 8
    DOP_RK8 = 8
    @property
    def steps(self):
        if self is RKMethod.EULER:
            return 1
        elif self is RKMethod.RK4:
            return 4
        elif self is RKMethod.SSP_RK3:
            return 3
        elif self is RKMethod.DOP_RK8:
            return 12
        else:
            raise Exception()

class WENOSolution(SolutionBase):
    """
    Interface class to define some functions that WENO solutions should have.
    """
    def set_rk_method(self, method):
        raise NotImplementedError()
    def set_record_actions(self, mode):
        raise NotImplementedError()
    def get_action_history(self):
        raise NotImplementedError()


# Should be possible to convert this to ND instead of 2D.
# rk_substep needs to be generalized.
class PreciseWENOSolution2D(WENOSolution):
    def __init__(self, base_grid, init_params,
            precise_order, precise_scale,
            flux_function, eqn_type='burgers',
            vec_len=1, reconstruction='componentwise', nu=0.0, source=None,
            record_state=False, record_actions=None, rk_method=RKMethod.EULER):
        super().__init__(base_grid.num_cells, base_grid.num_ghosts,
                base_grid.min_value, base_grid.max_value, vec_len, record_state=record_state)

        assert (precise_scale % 2 == 1), "Precise scale must be odd for easier downsampling."
        assert (vec_len == 1), "PreciseWENOSolution2D does not currently support vectors."

        self.precise_scale = precise_scale

        self.precise_num_cells = [precise_scale * nx for nx in base_grid.num_cells]
        self.precise_num_ghosts = []
        self.extra_ghosts = []
        for ng in base_grid.num_ghosts:
            if precise_order + 1 > precise_scale * ng:
                precise_ng = precise_order + 1
                self.precise_num_ghosts.append(precise_ng)
                self.extra_ghosts.append(precise_ng - (precise_scale*ng))
            else:
                self.precise_num_ghosts.append(precise_scale * ng)
                self.extra_ghosts.append(0)

        if 'boundary' in init_params:
            self.boundary = init_params['boundary']
        else:
            self.boundary = None
        if 'init_type' in init_params:
            self.init_type = init_params['init_type']
        else:
            self.init_type = None

        self.eqn_type = eqn_type
        self.reconstruction = reconstruction
        self.precise_grid = create_grid(len(self.precise_num_cells),
                self.precise_num_cells, self.precise_num_ghosts,
                base_grid.min_value, base_grid.max_value,
                init_type=self.init_type, boundary=self.boundary, eqn_type=eqn_type)

        self.flux_function = flux_function
        self.nu = nu
        self.precise_order = precise_order
        self.source = source
        self.rk_method = rk_method

        self.record_actions = record_actions
        self.action_history = []

    def is_recording_actions(self):
        return (self.record_actions is not None)
    def set_record_actions(self, record_mode):
        self.record_actions = record_mode
    def get_action_history(self):
        return self.action_history

    def rk_substep(self):
        g = self.precise_grid
        g.update_boundary()
        order = self.precise_order
        num_x, num_y = g.num_cells
        ghost_x, ghost_y = g.num_ghosts
        cell_size_x, cell_size_y = g.cell_size

        # compute flux at each point
        flux = self.flux_function(g.space)
        if self.eqn_type == 'burgers':
            (flux_left, flux_right), (flux_down, flux_up) = lf_flux_split_nd(flux, g.space, self.eqn_type)
        elif self.eqn_type == 'euler':
            (flux_left, flux_right), (flux_down, flux_up) = lf_flux_split_nd(flux, g.space, self.eqn_type, g.eos_gamma)
        else:
            raise NotImplementedError

        # Trim vertical ghost cells from horizontally split flux. (Should this be part of flux
        # splitting?)
        flux_left = flux_left[0, :, ghost_y:-ghost_y]  # TODO: change for vec length, now only using 0th dim -yiwei
        flux_right = flux_right[0, :, ghost_y:-ghost_y]
        right_stencil_indexes = create_stencil_indexes(stencil_size=order * 2 - 1,
                                                       num_stencils=num_x + 1,
                                                       offset=ghost_x - order)
        left_stencil_indexes = np.flip(right_stencil_indexes, axis=-1) + 1
        # Indexing is tricky here. I couldn't find a way without transposing and then transposing
        # back. (It might be possible, though.)
        left_stencils = (flux_left.transpose()[:, left_stencil_indexes]).transpose([1,0,2])
        right_stencils = (flux_right.transpose()[:, right_stencil_indexes]).transpose([1,0,2])
        if self.reconstruction == 'characteristicwise':
            boundary_state = (g.u[:, g.ng - 1:-g.ng] + g.u[:, g.ng:-g.ng + 1]) * 0.5
            levecs = levecs_vectorized(g, boundary_state)
            left_stencils = np.einsum('jli, jkl-> kli', left_stencils, levecs)
            right_stencils = np.einsum('jli, jkl-> kli', right_stencils, levecs)

        left_flux_reconstructed, left_weights = weno_reconstruct_nd(order, left_stencils)
        right_flux_reconstructed, right_weights = weno_reconstruct_nd(order, right_stencils)
        if self.reconstruction == 'componentwise':
            horizontal_flux_reconstructed = left_flux_reconstructed + right_flux_reconstructed
        if self.reconstruction == 'characteristicwise':
            revecs = revecs_vectorized(g, boundary_state)
            horizontal_flux_reconstructed = np.einsum('jil, il-> jl',
                                                      revecs, left_flux_reconstructed + right_flux_reconstructed)

        flux_down = flux_down[0, ghost_x:-ghost_x, :]
        flux_up = flux_up[0, ghost_x:-ghost_x, :]
        up_stencil_indexes = create_stencil_indexes(stencil_size=order * 2 - 1,
                                                    num_stencils=num_y + 1,
                                                    offset=ghost_y - order)
        down_stencil_indexes = np.flip(up_stencil_indexes, axis=-1) + 1
        up_stencils = flux_up[:, up_stencil_indexes]
        down_stencils = flux_down[:, down_stencil_indexes]
        if self.reconstruction == 'characteristicwise':
            boundary_state = (g.u[:, g.ng - 1:-g.ng] + g.u[:, g.ng:-g.ng + 1]) * 0.5
            levecs = levecs_vectorized(g, boundary_state)
            up_stencils = np.einsum('jli, jkl-> kli', up_stencils, levecs)
            down_stencils = np.einsum('jli, jkl-> kli', down_stencils, levecs)

        down_flux_reconstructed, down_weights = weno_reconstruct_nd(order, down_stencils)
        up_flux_reconstructed, up_weights = weno_reconstruct_nd(order, up_stencils)
        if self.reconstruction == 'componentwise':
            vertical_flux_reconstructed = up_flux_reconstructed + down_flux_reconstructed
        if self.reconstruction == 'characteristicwise':
            revecs = revecs_vectorized(g, boundary_state)
            vertical_flux_reconstructed = np.einsum('jil, il-> jl',
                                                    revecs, up_flux_reconstructed + down_flux_reconstructed)

        if self.record_actions is not None:
            if self.record_actions == "weno":
                # Changed this without testing, was the original version right?
                #self.action_history.append((np.stack([left_weights, right_weights], axis=-2),
                                                #np.stack([down_weights, up_weights], axis=-1)))
                self.action_history.append((np.stack([left_weights, right_weights], axis=2),
                                                np.stack([down_weights, up_weights], axis=2)))
            elif self.record_actions == "coef":
                raise NotImplementedError()
                # This corresponds to e.g. SplitFluxBurgersEnv.
                # Still need to convert this to 2D.
                # Related to Standard WENO agent in agents.py.
                #a_mat = weno_coefficients.a_all[order]
                #a_mat = np.flip(a_mat, axis=-1)
                #combined_weights = a_mat[None, None, :, :] * action_weights[:, :, :, None]

                #flux_weights = np.zeros((g.real_length() + 1, 2, self.order * 2 - 1))
                #for sub_stencil_index in range(order):
                    #flux_weights[:, :, sub_stencil_index:sub_stencil_index + order] += combined_weights[:, :, sub_stencil_index, :]
                #self.action_history.append(flux_weights)
            else:
                raise Exception("Unrecognized action type: '{}'".format(self.record_actions))

        rhs = (  (horizontal_flux_reconstructed[:-1, :]
                    - horizontal_flux_reconstructed[1:, :]) / cell_size_x
                + (vertical_flux_reconstructed[:, :-1]
                    - vertical_flux_reconstructed[:, 1:]) / cell_size_y
                )

        if self.nu > 0.0:
            rhs = rhs + self.nu * self.precise_grid.laplacian()
        if self.source is not None:
            rhs = rhs + self.source.get_real()

        return rhs

    def _update(self, dt, time):
        u_start = np.array(self.precise_grid.get_real())
        if self.rk_method is RKMethod.EULER:
            full_step = dt * self.rk_substep()
            self.precise_grid.set(u_start + full_step)
        elif self.rk_method is RKMethod.RK4:
            k1 = dt * self.rk_substep()
            self.precise_grid.set(u_start + (k1 / 2))

            k2 = dt * self.rk_substep()
            self.precise_grid.set(u_start + (k2 / 2))

            k3 = dt * self.rk_substep()
            self.precise_grid.set(u_start + k3)

            k4 = dt * self.rk_substep()
            full_step = (k1 + 2*(k2 + k3) + k4) / 6
            self.precise_grid.set(u_start + full_step)
        elif self.rk_method is RKMethod.SSP_RK3:
            u1 = u_start + dt * self.rk_substep()
            self.precise_grid.set(u1)

            u2 = (3 * u_start + u1 + dt * self.rk_substep()) / 4
            self.precise_grid.set(u2)

            u3 = (u_start + 2 * u2 + 2 * dt * self.rk_substep()) / 3
            self.precise_grid.set(u3)
        else:
            raise ValueError(f"{self.rk_method} RK method not recognized.")

        self.precise_grid.update_boundary()

    def get_full(self):
        """ Downsample the precise solution to get coarse solution values. """

        # Note that axis 0 is the vector dimension.

        grid = self.precise_grid.get_full()

        if any([xg > 0 for xg in self.extra_ghosts]):
            extra_ghost_slice = tuple([slice(None)] + [slice(xg, -xg) for xg in self.extra_ghosts])
            grid = grid[extra_ghost_slice]

        # For each coarse cell, there are precise_scale precise cells, where the middle
        # cell corresponds to the same location as the coarse cell.
        middle = int(self.precise_scale / 2)
        middle_cells_slice = tuple([slice(None)] + [slice(middle, None, self.precise_scale) for _ in
            self.num_cells])
        return grid[middle_cells_slice]

    def get_real(self):
        return self.get_full()[self.real_slice]

    def _reset(self, init_params):
        self.precise_grid.reset(init_params)
        self.action_history = []

    def set(self, real_values):
        """
        Force set the current grid.

        This may make the state/action history confusing.

        Also updates the boundary.
        """
        self.precise_grid.set(real_values)


class PreciseWENOSolution(WENOSolution):

    def __init__(self,
                 base_grid, init_params,
                 precise_order, precise_scale, flux_function, eqn_type='burgers',
                 vec_len=1, reconstruction='componentwise', nu=0.0, source=None,
                 record_state=False, record_actions=None, rk_method=RKMethod.EULER):
        super().__init__(base_grid.num_cells, base_grid.num_ghosts,
                base_grid.min_value, base_grid.max_value, vec_len, record_state=record_state)

        assert (precise_scale % 2 == 1), "Precise scale must be odd for easier downsampling."

        self.precise_scale = precise_scale

        self.precise_nx = precise_scale * base_grid.nx
        if precise_order + 1 > precise_scale * base_grid.ng:
            self.precise_ng = precise_order + 1
            self.extra_ghosts = self.precise_ng - (precise_scale * base_grid.ng)
        else:
            self.precise_ng = precise_scale * base_grid.ng
            self.extra_ghosts = 0

        if 'boundary' in init_params:
            self.boundary = init_params['boundary']
        else:
            self.boundary = None
        if 'init_type' in init_params:
            self.init_type = init_params['init_type']
        else:
            self.init_type = None

        self.eqn_type = eqn_type
        self.reconstruction = reconstruction
        self.precise_grid = create_grid(self.ndim,
                self.precise_nx, self.precise_ng,
                base_grid.min_value, base_grid.max_value,
                init_type=self.init_type, boundary=self.boundary, eqn_type=self.eqn_type)

        self.flux_function = flux_function
        self.nu = nu
        self.order = precise_order
        self.source = source
        self.rk_method = rk_method

        self.record_actions = record_actions
        self.action_history = []
        self.previous_action = None

    def is_recording_actions(self):
        return (self.record_actions is not None)
    def set_record_actions(self, record_mode):
        self.record_actions = record_mode
    def get_action_history(self):
        return self.action_history

    def rk_substep(self):

        # get the solution data
        g = self.precise_grid
        g.update_boundary()
        order = self.order

        # compute flux at each point
        f = self.flux_function(g.u)
        if self.eqn_type == 'burgers':
            flux_minus, flux_plus = lf_flux_split_nd(f, g.u, self.eqn_type)
        elif self.eqn_type == 'euler':
            flux_minus, flux_plus = lf_flux_split_nd(f, g.u, self.eqn_type, g.eos_gamma)
        else:
            raise NotImplementedError

        plus_stencil_indexes = create_stencil_indexes(stencil_size=order * 2 - 1,
                                                       num_stencils=g.nx + 1,
                                                       offset=g.ng - order)
        minus_stencil_indexes = np.flip(plus_stencil_indexes, axis=-1) + 1
        plus_stencils = flux_plus[:, plus_stencil_indexes]
        minus_stencils = flux_minus[:, minus_stencil_indexes]
        if self.reconstruction == 'characteristicwise':
            boundary_state = (g.u[:, g.ng - 1:-g.ng] + g.u[:, g.ng:-g.ng + 1]) * 0.5
            levecs = levecs_vectorized(g, boundary_state)
            plus_stencils = np.einsum('jli, jkl-> kli', plus_stencils, levecs)
            minus_stencils = np.einsum('jli, jkl-> kli', minus_stencils, levecs)

        plus_reconstructed, plus_weights = weno_reconstruct_nd(order, plus_stencils)
        minus_reconstructed, minus_weights = weno_reconstruct_nd(order, minus_stencils)
        if self.reconstruction == 'componentwise':
            flux_reconstructed = minus_reconstructed + plus_reconstructed
        if self.reconstruction == 'characteristicwise':
            revecs = revecs_vectorized(g, boundary_state)
            flux_reconstructed = np.einsum('jil, il-> jl', revecs, minus_reconstructed + plus_reconstructed)

        if self.record_actions is not None:
            # Weights have shape vec X location X sub-stencil.
            # Stack on axis 2 so resulting aray has shape
            # vec X location X (+,-) X sub-stencil.
            action_weights = np.stack([plus_weights, minus_weights], axis=2)

            if self.record_actions == "weno":
                self.previous_action = action_weights
            elif self.record_actions == "coef":
                order = self.order
                a_mat = weno_coefficients.a_all[order]
                a_mat = np.flip(a_mat, axis=-1)

                # a_mat is sub-stencil X inner_index.
                # action_weights is vec X location X (+,-) X sub-stencil.
                # We want vec X location X (+,-) X inner_index.
                combined_weights = a_mat[None, None, None, :, :] * action_weights[:, :, :, :, None]

                flux_weights = np.zeros((g.vec_len, g.real_length() + 1, 2, self.order * 2 - 1))
                # TODO: figure out a way to vectorize this. This is a weird broadcast,
                # which would be easier if numpy had builtin support for banded matrices.
                for sub_stencil_index in range(order):
                    i = sub_stencil_index
                    flux_weights[:, :, :, i:i + order] += combined_weights[:, :, :, i, :]
                self.previous_action = flux_weights
            else:
                raise Exception("Unrecognized action type: '{}'".format(self.record_actions))

        rhs = (flux_reconstructed[:, :-1] - flux_reconstructed[:, 1:]) / g.dx

        if self.nu > 0.0:
            rhs += self.nu * self.precise_grid.laplacian()
        if self.source is not None:
            rhs += self.source.get_real()

        return rhs

    def _update(self, dt, time):
        u_start = np.array(self.precise_grid.get_real())
        if self.rk_method is RKMethod.EULER:
            full_step = dt * self.rk_substep()
            self.precise_grid.set(u_start + full_step)
            if self.record_actions:
                self.action_history.append(self.previous_action)
        elif self.rk_method is RKMethod.RK4:
            k1 = dt * self.rk_substep()
            self.precise_grid.set(u_start + (k1 / 2))
            # Only record the first stage action.
            if self.record_actions:
                self.action_history.append(self.previous_action)

            k2 = dt * self.rk_substep()
            self.precise_grid.set(u_start + (k2 / 2))

            k3 = dt * self.rk_substep()
            self.precise_grid.set(u_start + k3)

            k4 = dt * self.rk_substep()
            full_step = (k1 + 2*(k2 + k3) + k4) / 6
            self.precise_grid.set(u_start + full_step)
        elif self.rk_method is RKMethod.SSP_RK3:
            u1 = u_start + dt * self.rk_substep()
            self.precise_grid.set(u1)
            if self.record_actions:
                self.action_history.append(self.previous_action)

            u2 = (3 * u_start + u1 + dt * self.rk_substep()) / 4
            self.precise_grid.set(u2)

            u3 = (u_start + 2 * u2 + 2 * dt * self.rk_substep()) / 3
            self.precise_grid.set(u3)
        else:
            raise ValueError(f"{self.rk_method} RK method not recognized.")

        self.precise_grid.update_boundary()

    def get_full(self):
        """ Downsample the precise solution to get coarse solution values. """

        # Note that axis 0 is the vector dimension.

        grid = self.precise_grid.get_full()
        if self.extra_ghosts > 0:
            grid = grid[:, self.extra_ghosts:-self.extra_ghosts]

        # For each coarse cell, there are precise_scale precise cells, where the middle
        # cell corresponds to the same location as the coarse cell.
        middle = int(self.precise_scale / 2)
        return grid[:, middle::self.precise_scale]

    def get_real(self):
        return self.get_full()[:, self.ng:-self.ng]

    def _reset(self, init_params):
        self.precise_grid.reset(init_params)
        self.action_history = []

    def set(self, real_values):
        """
        Force set the current grid.

        This may make the state/action history confusing.

        Also updates the boundary.
        """
        self.precise_grid.set(real_values)

    # These functions are not used. This was the original method for computing WENO. It corresponds
    # more directly to the original mathematical formulation, but is slower and not vectorized.
    # We're keeping these here for reference.
    def weno_stencils(self, q):
        """
        Compute WENO stencils

        Parameters
        ----------
        q : np array
          Scalar data to reconstruct.

        Returns
        -------
        stencils
        """
        order = self.order
        a = weno_coefficients.a_all[order]
        num_points = q.shape[1] - 2 * order  # len(q) changed to q.shape[1] for new 0-th dimension vec state
        q_stencils_all = []
        for nv in range(q.shape[0]):
            q_stencils = np.zeros((order, q.shape[1]))
            for i in range(order, num_points + order):
                for k in range(order):
                    for l in range(order):
                        q_stencils[k, i] += a[k, l] * q[nv, i + k - l]
            q_stencils_all.append(q_stencils)

        return np.array(q_stencils_all)
    def weno_weights(self, q):
        """
        Compute WENO weights

        Parameters
        ----------
        q : np array
          Scalar data to reconstruct.

        Returns
        -------
        stencil weights
        """
        order = self.order
        C = weno_coefficients.C_all[order]
        sigma = weno_coefficients.sigma_all[order]

        beta = np.zeros((order, q.shape[1]))
        w = np.zeros_like(beta)
        num_points = q.shape[1] - 2 * order
        epsilon = 1e-16
        w_all = []
        for nv in range(q.shape[0]):
            for i in range(order, num_points + order):
                alpha = np.zeros(order)
                for k in range(order):
                    for l in range(order):
                        for m in range(l + 1):
                            beta[k, i] += sigma[k, l, m] * q[nv, i + k - l] * q[nv, i + k - m]
                    alpha[k] = C[k] / (epsilon + beta[k, i] ** 2)
                    # TODO: why the reconstruction is different than below Euler implementation? Double check. -yiwei
                    # https://github.com/python-hydro/hydro_examples/blob/master/compressible/weno_euler.py#L97
                w[:, i] = alpha / np.sum(alpha)
            w_all.append(w)

        return np.array(w_all)
    def weno_new(self, q):
        """
        Compute WENO reconstruction

        Parameters
        ----------
        q : numpy array
          Scalar data to reconstruct.

        Returns
        -------
        qL: numpy array
          Reconstructed data.

        """
        weights = self.weno_weights(q)
        q_stencils = self.weno_stencils(q)
        qL = np.zeros_like(q)
        num_points = q.shape[1] - 2 * self.order
        for nv in range(q.shape[0]):
            for i in range(self.order, num_points + self.order):
                qL[nv, i] = np.dot(weights[nv, :, i], q_stencils[nv, :, i])
        return qL, weights

if __name__ == "__main__":
    import os
    import shutil
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    from matplotlib import cm
    from matplotlib.ticker import LinearLocator
    from envs.grid2d import Burgers2DGrid

    order = 2
    ng = order + 1

    base_grid = Burgers2DGrid((128, 128), ng, 0.0, 1.0)
    flux_function = lambda x: 0.5 * x ** 2
    sol = PreciseWENOSolution2D(base_grid, {}, order, 1, flux_function)
    #sol.reset({'init_type': '1d-accelshock-x'})
    sol.reset({'init_type': 'gaussian'})

    timestep = 0.0004
    time = 0.0

    save_dir = "weno2d_test"
    try:
        os.makedirs(save_dir)
    except FileExistsError:
        # Directory already exists, overwrite.
        shutil.rmtree(save_dir)
        os.makedirs(save_dir)

    steps = 500

    for step in range(steps + 1):
        if step % 10 == 0:
            print("{}...".format(step), end='', flush=True)
            fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
            x, y = np.meshgrid(sol.real_x, sol.real_y, indexing='ij')
            z = sol.get_real()
            surface = ax.plot_surface(x, y, z[0], cmap=cm.viridis,
                    linewidth=0, antialiased=False)
            #ax.set_zlim(-0.25, 1.25)
            ax.zaxis.set_major_locator(LinearLocator(10))
            #ax.zaxis.set_major_formatter(StrMethodFormatter('{x:0.2f}'))
            fig.colorbar(surface, shrink=0.5, aspect=5)

            ax.set_title("t={:.4f}".format(time))

            filename = os.path.join(save_dir, "plot_{:03d}.png".format(step))
            plt.savefig(filename)
            plt.close(fig)
        time += timestep
        sol.update(timestep, time)
    print("Done")
