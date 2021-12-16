# Originally adapted from code for
# Introduction to Computational Astrophysical Hydrodynamics
# by M. Zingale (2013-03-26).

import numpy as np
import tensorflow as tf

from util.misc import AxisSlice, TensorAxisSlice


class AbstractGrid:
    """
    Abstract base class for Grids.

    An AbstractGrid represents a quantity in a physical space that has been discretized
    into cells. An AbstractGrid contains real cells, representing the values within the
    physical space, and ghost cells, representing values just beyond the
    boundary of the physical space necessary for computing convolutions over
    the phyiscal space.

    A grid can be reset to a state specified by a standard parameter dict,
    and set using a list of real values (the ghost cells should be updated
    using some other means).

    How the AbstractGrid represents the physical space is up to the subclass. It may not have any
    persistent representation, as with an analytical solution, or it may contain a separate
    AbstractGrid itself.
    """

    def __init__(self, num_cells, num_ghosts, min_value, max_value, vec_len):
        """
        Construct a new Grid of arbitrary physical dimensions.

        num_cells should be an iterable defining the number of cells in the grid, e.g. [5,3,7] for
        a 5x3x7 grid. num_cells can be a scalar e.g. 5 for a 1-dimensional grid.
        num_ghosts, min_value, and max_value can each be iterables of the same size as num_cells,
        giving specific values for each dimension, or scalars, giving the same value for each
        dimension (useful for square grids).

        Parameters
        ---------
        num_cells : [int] OR int
            List of the number of cells with which to discretize each dimension.
            Can be a single int if the grid is one dimensional.
        num_ghosts : [int] OR int
            The number of ghost cells for each dimension. Either a list that gives the number of
            each dimension, or a single int to use the same number for each dimension.
        min_value : [float] OR float
            The lower bound of the grid for each dimension. Either a list that gives the lower
            bound for each dimension, or a single float to give each dimension the same lower
            bound.
        max_value : [float] OR float
            The upper bound of the grid for each dimension. Either a list that gives the upper
            bound for each dimension, or a single float to give each dimension the same upper
            bound.
        vec_len : int
            Length of the state vector, corresponding to the first dimension of grid.space
        """

        try:
            iterator = iter(num_cells)
        except TypeError:
            self.one_dimensional = True
            self.num_cells = (num_cells,)
        else:
            self.one_dimensional = False
            self.num_cells = num_cells
        
        try:
            if len(num_ghosts) != len(self.num_cells):
                raise ValueError("AbstractGrid: Size of num_ghosts must match size of num_cells"
                        + " ({} vs {}).".format(len(num_ghosts), len(self.num_cells)))
            else:
                self.num_ghosts = num_ghosts
        except TypeError:
            self.num_ghosts = (num_ghosts,) * len(self.num_cells)

        try:
            if len(min_value) != len(self.num_cells):
                raise ValueError("AbstractGrid: Size of min_value must match size of num_cells"
                        + " ({} vs {}).".format(len(min_value), len(self.num_cells)))
            else:
                self.min_value = min_value
        except TypeError:
            self.min_value = (min_value,) * len(self.num_cells)

        try:
            if len(max_value) != len(self.num_cells):
                raise ValueError("AbstractGrid: Size of max_value must match size of num_cells"
                        + " ({} vs {}).".format(len(max_value), len(self.num_cells)))
            else:
                self.max_value = max_value
        except TypeError:
            self.max_value = (max_value,) * len(self.num_cells)

        self.vec_len = vec_len

        self.cell_size = []
        # Physical coordinates: cell-centered
        # Note that for >1 dimension, these are not the coordinates themselves - 
        # the actual coordinates are the cross product of these,
        # e.g. coords[0] X coords[1] X coords[2].
        self.coords = []
        # cell-centered coordinates, only real cells (not ghosts)
        self.real_coords = []
        # Physical coordinates of the interfaces (including min and max edges).
        # Note that these are also not the coordinates of the interfaces.
        # Substitute these into the cross product of coords to get interfaces along each dimension,
        # so e.g. coords[0] X interfaces[1] X coords[2] to get the interfaces between cells that
        # are adjacent on the 1 dimension.
        self.interfaces = []

        for nx, ng, xmin, xmax in zip(
                self.num_cells, self.num_ghosts, self.min_value, self.max_value):
            dx = (xmax - xmin) / nx
            self.cell_size.append(dx)

            x = xmin + (np.arange(nx + 2 * ng) - ng + 0.5) * dx
            self.coords.append(x)
            real_x = x[ng:-ng]
            self.real_coords.append(real_x)
            inter_x = xmin + (np.arange(nx + 2 * ng + 1) - ng) * dx
            self.interfaces.append(inter_x)

        self.real_slice = tuple([slice(0, self.vec_len)] + [slice(ng, -ng) for ng in self.num_ghosts])
        """
        Slice for indexing only the real portion and not ghost cells of the space (or something
        of equivalent shape).
        So grid.space[grid.real_slice] should be equivalent to grid.get_real().
        0-th dimension is the vector length dim, all needs to be in
        """

    # 1-dimensional shortcuts for compatability.
    @property
    def nx(self): return self.num_cells[0]
    @property
    def ng(self): return self.num_ghosts[0]
    @property
    def xmin(self): return self.min_value[0]
    @property
    def xmax(self): return self.max_value[0]
    @property
    def inter_x(self): return self.interfaces[0]
    @property
    def dx(self): return self.cell_size[0]

    # x, y, and z can be more readable.
    @property
    def x(self): return self.coords[0]
    @property
    def real_x(self): return self.real_coords[0]
    @property
    def inter_x(self): return self.interfaces[0]
    @property
    def y(self): return self.coords[1]
    @property
    def real_y(self): return self.real_coords[1]
    @property
    def inter_y(self): return self.interfaces[1]
    @property
    def z(self): return self.coords[2]
    @property
    def real_z(self): return self.real_coords[2]
    @property
    def inter_z(self): return self.interfaces[2]
    
    #TODO If grids become really big, then we won't be able to read/write them all at once.
    # Implement __get_item__ and __set_item__ (i.e. the [] operator) if that happens.
    def set(self, real_values):
        raise NotImplementedError()

    def get_real(self):
        raise NotImplementedError()

    def get_full(self):
        raise NotImplementedError()

    def reset(self, params_dict):
        raise NotImplementedError()

    def scratch_array(self):
        """ Return a zeroed array dimensioned for this grid. """
        space = np.zeros([self.vec_len] + [len(x) for x in self.coords], dtype=np.float64)
        # if len(space) == 1:
        #     space = space[0]  # For backward compatibility, scalar state doesn't need 1st dim
        return space

    @property
    def ndim(self): return len(self.num_cells)

class GridBase(AbstractGrid):
    """
    Slightly less abstract Grid class.

    A GridBase uses a Numpy ndarray to represent the phyiscal space. This allows for default
    implementations of set(), get_real() and get_full().

    The reset() method must still be implemented in the subclass, as the potential varieties of
    parameters with which to initialize the grid typically depend on the dimension.

    set() calls self.update_boundary(). update_boundary() should be overriden in a base class if
    other boundary conditions are required.
    """

    # Default schedule for the 'schedule' init. Define in a concrete subclass to be compatible with
    # the get_default_schedule() function.
    DEFAULT_SCHEDULE = None

    def __init__(self, num_cells, num_ghosts, min_value, max_value, vec_len, boundary="outflow"):
        super().__init__(num_cells, num_ghosts, min_value, max_value, vec_len)
     
        if type(boundary) is str:
            if len(self.num_cells) == 1:
                self.boundary = boundary
            else:
                self.boundary = (boundary,) * len(self.num_cells)
            """
            GridBase.boundary is one of these annoying properties that can be either a tuple or a
            string. The string implies (str,) * len(self.num_cells), but for compability reasons
            you should check which type it is before using it.
            """
        else:
            if boundary is not None and len(boundary) != len(self.num_cells):
                raise ValueError("GridBase: Size of boundary must match size of num_cells"
                        + " ({} vs {}).".format(len(self.boundary), len(self.num_cells)))
            else:
                self.boundary = boundary

        # Storage for the solution.
        self.space = self.scratch_array()
        """ The physical space. """

    # Old names for compatability.
    @property
    def u(self): return self.space
    @u.setter
    def u(self, value): self.space = value

    def set(self, new_values):
        """
        Set the real (non-ghost) values in the grid.

        The ghost cells are also updated by internally calling self.update_boundary().

        Parameters
        ---------
        new_values : array-like
            Array of new values. The values should map to real cells and not ghost cells,
            so len(new_values) == grid.num_cells.
        """
        self.space[self.real_slice] = new_values
        self.update_boundary()

    def get_real(self):
        """
        Get the real (non-ghost) values in the grid.
        Note that this returns a WRITABLE view on the internal ndarray.
        """
        return self.space[self.real_slice]
    def get_full(self):
        """
        Get the full grid, including ghost cells.
        Note that this returns a WRITABLE view on the internal ndarray.
        """
        return self.space

    def update_boundary(self):
        """
        Update the ghost cells based on the value of the grid.boundary field.

        Grid.set calls this method, so you need only use this method if accessing the grid by some
        other means, such as writing directly to grid.space.
        """
        boundary = self.boundary
        if boundary is None:
            raise Exception("GridBase: boundary must be set before update_boundary is called.")
        if type(boundary) is str:
            boundary = (boundary,) * len(self.num_cells)
        elif len(boundary) != len(self.num_cells):
            raise ValueError("GridBase: Size of boundary must match size of num_cells"
                    + " ({} vs {}).".format(len(boundary), len(self.num_cells)))

        for axis, (ng, bound) in enumerate(zip(self.num_ghosts, boundary)):
            axis += 1  # With the current implementation, the 0-th dimension is taken as the vector length dim
            axis_slice = AxisSlice(self.space, axis)
            # Periodic - copying from the opposite end, as if the space wraps around
            if bound == "periodic":
                    axis_slice[:ng] = axis_slice[-2*ng: -ng]
                    axis_slice[-ng:] = axis_slice[ng: 2*ng]
            # Outflow - extending the edge values past the boundary
            elif bound == "outflow":
                    axis_slice[:ng] = axis_slice[ng]
                    axis_slice[-ng:] = axis_slice[-ng - 1]
            else:
                raise Exception("GridBase: Boundary type \"" + str(bound) + "\" not recognized.")

    @tf.function
    def tf_update_boundary(self, real_state, boundary=None):
        """
        Equivalent to grid.update_boundary(), except as a Tensorflow function that can be applied
        to an arbitrary grid with arbitrary boundaries.

        Returns the state extended with ghost cells.
        """
        if boundary is None:
            if self.boundary is None:
                raise Exception("GridBase: default boundary is None.")
            else:
                boundary = self.boundary

        if type(boundary) is str:
            boundary = (boundary,) * len(self.num_cells)
        elif len(boundary) != len(self.num_cells):
            raise ValueError("GridBase: Size of boundary must match size of num_cells"
                    + " ({} vs {}).".format(len(boundary), len(self.num_cells)))

        filled_state = real_state

        for axis, (ng, bound) in enumerate(zip(self.num_ghosts, boundary)):
            axis = axis + 1 # axis 0 is the vector axis
            axis_slice = TensorAxisSlice(filled_state, axis)
            if bound == "periodic":
                left_ghost = axis_slice[-ng:]
                right_ghost = axis_slice[:ng]
                filled_state = tf.concat([left_ghost, filled_state, right_ghost], axis=axis)
            elif bound == "outflow":
                tile_multiples = [1] * (self.ndim + 1)
                tile_multiples[axis] = ng
                left_ghost = tf.tile(axis_slice[:1], tile_multiples)
                right_ghost = tf.tile(axis_slice[-1:], tile_multiples)
                filled_state = tf.concat([left_ghost, filled_state, right_ghost], axis=axis)
            else:
                raise Exception("GridBase: Boundary type \"" + str(bound) + "\" not recognized.")

        return filled_state

    def laplacian(self):
        """
        Compute the Laplacian of the current grid.

        Returns an ndarray of the same shape as the real space.
        """
        partial_2nd_derivatives = []


        for axis, (ng, dx) in enumerate(zip(self.num_ghosts, self.cell_size)):
            axis = axis + 1 # Axis 0 is the vector axis.
            # Slice away ghost values on all but the relevant axis.
            mostly_real_slice = list(self.real_slice)
            mostly_real_slice[axis] = slice(None)
            mostly_real_slice = tuple(mostly_real_slice)
            mostly_real_space = self.space[mostly_real_slice]
            axis_slice = AxisSlice(mostly_real_space, axis)
            d2fdx2 = (axis_slice[ng - 1:-(ng + 1)] 
                    - 2.0 * axis_slice[ng:-ng] 
                    + axis_slice[ng + 1:-(ng - 1)]) / dx**2
            partial_2nd_derivatives.append(d2fdx2)

        return np.sum(partial_2nd_derivatives, axis=0)

    @tf.function
    def tf_laplacian(self, real_state):
        """
        Unlike grid.laplacian(), this takes a state portioned as the REAL state and computes ghost
        cells on the fly. (The Laplacian only needs the first cell past the boundary.)
        """
        boundary = self.boundary
        if type(boundary) is str:
            boundary = (boundary,) * len(self.num_cells)
        elif len(boundary) != len(self.num_cells):
            raise ValueError("GridBase: Size of boundary must match size of num_cells"
                    + " ({} vs {}).".format(len(boundary), len(self.num_cells)))

        partial_2nd_derivatives = []
        for axis, (dx, bound) in enumerate(zip(self.cell_size, boundary)):
            axis = axis + 1 # Axis 0 is the vector axis.
            axis_slice = TensorAxisSlice(real_state, axis)
            central_lap = (axis_slice[:-2] - 2.0*axis_slice[1:-1] + axis_slice[2:]) / (dx**2)
            if bound == "outflow":
                left_lap = (-axis_slice[0] + axis_slice[1]) / (dx**2) # X-2X+Y = -X+Y
                right_lap = (axis_slice[-2] - axis_slice[-1]) / (dx**2) # X-2Y+Y = X-Y
            elif bound == "periodic":
                left_lap = (axis_slice[-1] - 2.0*axis_slice[0] + axis_slice[1]) / (dx**2)
                right_lap = (axis_slice[-2] - 2.0*axis_slice[-1] + axis_slice[0]) / (dx**2)
            else:
                raise NotImplementedError()
            d2fdx2 = tf.concat([[left_lap], central_lap, [right_lap]], axis=axis)
            partial_2nd_derivatives.append(d2fdx2)

        return tf.reduce_sum(partial_2nd_derivatives, axis=0)

def _is_list(thing):
    try:
        _iterator = iter(thing)
    except TypeError:
        return True
    else:
        return False

# Import down here to avoid circular import.
from envs.grid1d import Burgers1DGrid, Euler1DGrid
from envs.grid2d import Burgers2DGrid

def create_grid(num_dimensions, num_cells, num_ghosts, min_value, max_value, eqn_type='burgers',
        boundary=None, init_type=None, schedule=None,
        deterministic_init=False):
    if eqn_type == 'burgers':
        if num_dimensions == 1:
            return Burgers1DGrid(num_cells, num_ghosts, min_value, max_value,
                                 init_type=init_type, boundary=boundary, schedule=schedule,
                                 deterministic_init=deterministic_init)
        elif num_dimensions == 2:
            return Burgers2DGrid(num_cells, num_ghosts, min_value, max_value,
                                 init_type=init_type, boundary=boundary, schedule=schedule,
                                 deterministic_init=deterministic_init)
        else:
            raise NotImplementedError()
    elif eqn_type == 'euler':
        if num_dimensions == 1:
            return Euler1DGrid(num_cells, num_ghosts, min_value, max_value,
                               init_type=init_type, boundary=boundary, schedule=schedule,
                               deterministic_init=deterministic_init)
        else:
            raise NotImplementedError()
    else:
        raise NotImplementedError()


def get_default_schedule(env_name):
    """
    Find the default schedule for the 'schedule' initial condition based on the environment name.

    This is found in each concrete Grid class, and each environment name has a Grid class
    associated with it.
    """
    # Kind of a hack - I'm creating this function so batch size can default properly to include at
    # least every init in the schedule.
    if env_name == "weno_burgers":
        return Burgers1DGrid.DEFAULT_SCHEDULE
    elif env_name == "weno_burgers_2d":
        return Burgers2DGrid.DEFAULT_SCHEDULE
    elif env_name == "weno_euler":
        return Euler1DGrid.DEFAULT_SCHEDULE
    else:
        raise Exception(f"Need to extend 'get_default_schedule()' for {env_name}.")
