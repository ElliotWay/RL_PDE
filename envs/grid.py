# Originally adapted from code for
# Introduction to Computational Astrophysical Hydrodynamics
# by M. Zingale (2013-03-26).

import numpy as np

from util.misc import AxisSlice


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

    def __init__(self, num_cells, num_ghosts, min_value, max_value):
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

        self.real_slice = tuple([slice(ng, -ng) for ng in self.num_ghosts])
        """
        Slice for indexing only the real portion and not ghost cells of the space (or something
        of equivalent shape).
        So grid.space[grid.real_slice] should be equivalent to grid.get_real().
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
        return np.zeros([len(x) for x in self.coords], dtype=np.float64)

    @property
    def ndim(self): return len(self.num_cells)

class GridBase(AbstractGrid):
    """
    Slightly less abstract Grid class.

    A GridBase uses a Numpy ndarray to represent the phyiscal space. This allows for default
    implementations of set(), get_real() and get_full().

    The reset() method must still be implemented in the base class, as the potential varieties of
    parameters with which to initialize the grid typically depend on the dimension.

    set() calls self.update_boundary(). update_boundary() should be overriden in a base class if
    other boundary conditions are required.
    """

    def __init__(self, num_cells, num_ghosts, min_value, max_value, boundary="outflow"):
        super().__init__(num_cells, num_ghosts, min_value, max_value)
     
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

    # TODO Create a TF version of this.
    def laplacian(self):
        """
        Compute the Laplacian of the current grid.

        Returns an ndarray of the same shape as the real space.
        """
        partial_2nd_derivatives = []

        for axis, (ng, dx) in enumerate(zip(self.num_ghosts, self.cell_size)):
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


def _is_list(thing):
    try:
        _iterator = iter(thing)
    except TypeError:
        return True
    else:
        return False

# Import down here to avoid circular import.
from envs.grid1d import Grid1d
from envs.grid2d import Grid2d

def create_grid(num_dimensions, num_cells, num_ghosts, min_value, max_value,
        boundary=None, init_type=None,
        deterministic_init=False):
    if num_dimensions == 1:
        return Grid1d(num_cells, num_ghosts, min_value, max_value,
                init_type=init_type, boundary=boundary, deterministic_init=deterministic_init)
    elif num_dimensions == 2:
        return Grid2d(num_cells, num_ghosts, min_value, max_value,
                init_type=init_type, boundary=boundary, deterministic_init=deterministic_init)
    else:
        raise NotImplementedError()
