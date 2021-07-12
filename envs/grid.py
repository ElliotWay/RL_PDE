# The 1D version was originally adapted from code for
# Introduction to Computational Astrophysical Hydrodynamics
# by M. Zingale (2013-03-26).

import sys
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
        else:
            self.one_dimensional = False
        
        self.num_cells = num_cells

        if not self.one_dimensional:
            try:
                if len(num_ghosts) != len(num_cells):
                    raise ValueError("GridBase: Size of num_ghosts must match size of num_cells"
                            + " ({} vs {}).".format(len(num_ghosts), len(num_cells)))
                else:
                    self.num_ghosts = num_ghosts
            except TypeError:
                self.num_ghosts = (num_ghosts,) * len(num_cells)

            try:
                if len(min_value) != len(num_cells):
                    raise ValueError("GridBase: Size of min_value must match size of num_cells"
                            + " ({} vs {}).".format(len(min_value), len(num_cells)))
                else:
                    self.min_value = min_value
            except TypeError:
                self.min_value = (min_value,) * len(num_cells)

            try:
                if len(max_value) != len(num_cells):
                    raise ValueError("GridBase: Size of max_value must match size of num_cells"
                            + " ({} vs {}).".format(len(max_value), len(num_cells)))
                else:
                    self.max_value = max_value
            except TypeError:
                self.max_value = (max_value,) * len(num_cells)
        else:
            self.num_ghosts = num_ghosts
            self.min_value = min_value
            self.max_value = max_value

        if self.one_dimensional:
            self.dx = (self.max_value - self.min_value) / self.num_cells
            self.coords = (self.min_value + 
                (np.arange(self.num_cells + 2 * self.num_ghosts)
                        - self.num_ghosts + 0.5) * self.dx)
            self.real_coords = self.coords[self.num_ghosts:-self.num_ghosts]
            self.interfaces = (self.min_value +
                (np.arange(self.num_cells + 2 * self.num_ghosts)
                        - self.num_ghosts) * self.dx)
        else:
            # Cell size.
            self.dx = []
            # Physical coordinates: cell-centered, left and right edges
            # Note that for >1 dimension, these are not the coordinates themselves - 
            # the actual coordinates are the cross product of these,
            # e.g. coords[0] X coords[1] X coords[2].
            self.coords = []
            # cell-centered coordinates, only real cells (not ghosts)
            self.real_coords = []
            # Physical coordinates: interfaces, left edges
            self.interfaces = []

            for nx, ng, xmin, xmax in zip(
                    self.num_cells, self.num_ghosts, self.min_value, self.max_value):
                dx = (xmax - xmin) / nx
                self.dx.append(dx)

                x = xmin + (np.arange(nx + 2 * ng) - ng + 0.5) * dx
                self.coords.append(x)
                real_x = x[ng:-ng]
                self.real_coords.append(real_x)
                inter_x = xmin + (np.arange(nx + 2 * ng) - ng) * dx
                self.interfaces.append(inter_x)

    # Old names for compatability.
    @property
    def nx(self): return self.num_cells
    @property
    def ng(self): return self.num_ghosts
    @property
    def xmin(self): return self.min_value
    @property
    def xmax(self): return self.max_value
    @property
    def inter_x(self): return self.interfaces

    # x, y, and z make for more readable initial conditions.
    @property
    def x(self):
        if self.one_dimensional:
            return self.coords
        else:
            return self.coords[0]
    @property
    def real_x(self):
        if self.one_dimensional:
            return self.real_coords
        else:
            return self.real_coords[0]
    @property
    def y(self): return self.coords[1]
    @property
    def real_y(self): return self.real_coords[1]
    @property
    def z(self): return self.coords[2]
    @property
    def real_z(self): return self.real_coords[2]
    
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
        if self.one_dimensional:
            return np.zeros((self.nx + 2 * self.ng), dtype=np.float64)
        else:
            return np.zeros([len(x) for x in self.coords], dtype=np.float64)

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
        
        self.boundary = boundary

        # Storage for the solution.
        self.space = self.scratch_array()

        if self.one_dimensional:
            # slice(a,b) is equivalent to a:b.
            self.real_slice = slice(self.num_ghosts, -self.num_ghosts)
        else:
            self.real_slice = tuple([slice(ng, -ng) for ng in self.num_ghosts])

    # Old names for compatability.
    @property
    def u(self): return self.space

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
        # Periodic - copying from the opposite end, as if the space wraps around
        if self.one_dimensional:
            if self.boundary == "periodic":
                self.space[:self.num_ghosts] = self.space[-2*self.num_ghosts : -self.num_ghosts]
                self.space[-self.num_ghosts:] = self.space[self.num_ghosts : 2*self.num_ghosts]
            elif self.boundary == "outflow":
                self.space[:self.num_ghosts] = self.u[self.num_ghosts]
                self.space[-self.num_ghosts:] = self.u[-self.num_ghosts - 1]
            else:
                raise Exception("Boundary type \"" + str(self.boundary) + "\" not recognized.")

        else:
            #TODO: Could also loop over iterable boundary condition if we want different boundaries
            # for different axes.
            for axis, ng in enumerate(self.num_ghosts):
                axis_slice = AxisSlice(self.space, axis)
                if self.boundary == "periodic":
                        axis_slice[:ng] = axis_slice[-2*ng: -ng]
                        axis_slice[-ng:] = axis_slice[ng: 2*ng]
                elif self.boundary == "outflow":
                        axis_slice[:ng] = axis_slice[ng]
                        axis_slice[-ng:] = axis_slice[-ng - 1]
                else:
                    raise Exception("Boundary type \"" + str(self.boundary) + "\" not recognized.")

class Grid1d(GridBase):
    """
    1-dimensional grid of values for use in modelling differential equations.
    
    Fields intended for external use:
    grid.u - Current values in the grid, including ghost cells.
    Grid1d.update() updates the ghost cells automatically. If updating grid.u directly,
    remember to call update_boundary().
    grid.nx
    grid.ng
    grid.xmin
    grid.xmax
    grid.init_type - The current initialization, which can vary if using scheduled
    or sampled initialization.
    grid.boundary - The current boundary condition, which can vary similar to init_type.
    grid.init_params - All initialization parameters, including various numeric parameters
    as well as init_type and boundary. This is only set after reset() is called.
    grid.ilo
    grid.ihi
    grid.dx
    grid.x - The location values associated with each index in grid.u.
    """

    def __init__(self, nx, ng, xmin=0.0, xmax=1.0, init_type="sine", boundary="outflow", deterministic_init=False):
        """
        Construct a 1D grid.
        
        This also represents grid cells "beyond" the boundary.

        Parameters
        ----------
        
        nx : int
            Number of indexes in the discretized grid
        ng : int
            How many indexes beyond the boundary to include to represent
            boundary conditions. How many you need depends on your
            method of approximation.
        xmin : float
            coordinate of the beginning of the grid
        xmax : float
            coordinate of the end of the grid
        init_type : string
            type of initial condition
        boundary : string
            Type of boundary condition, either "periodic", "outflow",
            or None to choose based on the initial condition.
        deterministic_init : bool
            If False, the exact shape of the initial condition may vary.
            If True, it will always be exactly the same.
            "sample" will still sample randomly, but the sample initial
            condition will vary or be the same, depending on this param.
        """

        super().__init__(nx, ng, xmin, xmax, boundary)

        # _init_type and _boundary do not change, init_type and boundary may
        # change if init_type is scheduled or sampled.
        self._init_type = init_type
        self.init_type = init_type
        self._boundary = boundary
        self.boundary = boundary
        self.deterministic_init = deterministic_init
        self._init_schedule_index = 0
        #self._init_schedule = ["smooth_rare", "smooth_sine", "random", "rarefaction", "accelshock"]
        self._init_schedule = ["smooth_sine", "smooth_rare", "accelshock"]
        self._init_sample_types = self._init_schedule
        self._init_sample_probs = [1/len(self._init_sample_types)]*len(self._init_sample_types)

        # 0 and len-1 are indexes for values beyond the boundary,
        # so create ilo and ihi as indexes to real values
        self.ilo = ng
        self.ihi = ng + nx - 1

    def real_length(self):
        """ Return the number of indexes of real (non-ghost) points """
        return self.nx
    def full_length(self):
        """ Return the number of indexes of all points, including ghost points """
        return self.nx + 2 * self.ng

    def reset(self, params={}):
        """
        Reset the grid to initial conditions.

        Parameters
        ---------
        params : dict
            By default, reset based on the already set init_type and boundary fields,
        but you can pass a dict here to force it to reset to a specific state.
        The format of this dict is the same as the dict in init_params, which includes
        things like randomly generated numeric constants as well as the init_type.
        """
        if params is None:
            params = {}

        if 'init_type' in params:
            self.init_type = params['init_type']
        elif self._init_type == "schedule":
            self.init_type = self._init_schedule[self._init_schedule_index]
            self._init_schedule_index = (self._init_schedule_index + 1) % len(self._init_schedule)
        elif self._init_type == "sample":
            self.init_type = np.random.choice(self._init_sample_types, p=self._init_sample_probs)

        if 'boundary' in params:
            boundary = params['boundary']
        else:
            boundary = self._boundary
        self.boundary = boundary

        self.init_params = {'init_type': self.init_type}

        if self.init_type == "custom" or type(self.init_type) is not str:
            assert callable(self._init_type), "Custom init must have function provided as init type."
            assert boundary is not None, "Cannot use default boundary with custom init type."
            new_u, custom_params = self._init_type(params)
            self.init_params.update(custom_params)
            self.u[self.ng:-self.ng] = new_u

        elif self.init_type == "smooth_sine":
            if boundary is None:
                self.boundary = "periodic"
            if 'A' in params:
                A = params['A']
            else:
                A = float(1.0 / (2.0 * np.pi * 0.1))
            self.init_params['A'] = A
            self.u = A*np.sin(2 * np.pi * self.x)

        elif self.init_type == "gaussian":
            if boundary is None:
                self.boundary = "periodic"
            self.u = 1.0 + np.exp(-60.0 * (self.x - 0.5) ** 2)

        elif self.init_type == "random":
            if boundary is None:
                self.boundary = "periodic"
            if 'k' in params:
                k = params['k']
            elif self.deterministic_init:
                k = 4
            else:
                # Note that k must be even integer.
                k = int(np.random.choice(np.arange(2, 10, 2)))
            self.init_params['k'] = k
            if 'b' in params:
                b = params['b']
            elif self.deterministic_init:
                b = 0.5
            else:
                b = float(np.random.uniform(-1.0, 1.0))
            self.init_params['b'] = b
            if 'a' in params:
                a = params['a']
            else:
                a = float(3.5 - np.abs(b))
            self.init_params['a'] = a
            self.u = a + b * np.sin(k * np.pi * self.x / (self.xmax - self.xmin))

        elif self.init_type == "random-many-shocks":
            if boundary is None:
                self.boundary = "periodic"
            if 'k' in params:
                k = params['k']
            elif self.deterministic_init:
                k = 12 #6
            else:
                # Note that k must be even integer.
                k = int(np.random.choice(np.arange(10, 16, 2)))
            self.init_params['k'] = k
            if 'b' in params:
                b = params['b']
            elif self.deterministic_init:
                b = 0.5
            else:
                b_size = np.random.choice(np.linspace(0.2, 1.0, 9))
                b_sign = np.random.randint(2) * 2 - 1
                b = b_sign * b_size
            self.init_params['b'] = b
            if 'a' in params:
                a = params['a']
            elif self.deterministic_init:
                a = 0.0
            else:
                a = np.random.choice(np.linspace(-2.0, 2.0, 9))
            self.init_params['a'] = a
            self.u = a + b * np.sin(k * np.pi * self.x / (self.xmax - self.xmin))

        elif self.init_type == "smooth_rare":
            if boundary is None:
                self.boundary = "outflow"
            if 'A' in params:
                A = params['A']
            else:
                A = 1.0
            self.init_params['A'] = A
            if 'k' in params:
                k = params['k']
            elif self.deterministic_init:
                k = 60
            else:
                #k = np.random.uniform(20, 100)
                k = int(np.random.choice(np.arange(20, 100, 5)))
            self.init_params['k'] = k
            self.u = A * np.tanh(k * (self.x - 0.5))

        elif self.init_type == "accelshock":
            if boundary is None:
                self.boundary = "outflow"

            offset = params['offset'] if 'offset' in params else 0.25
            self.init_params['offset'] = offset
            u_L = params['u_L'] if 'u_L' in params else 3.0
            self.init_params['u_L'] = u_L
            u_R = params['u_R'] if 'u_R' in params else 3.0
            self.init_params['u_R'] = u_R

            index = self.x > offset
            self.u = np.full_like(self.x, u_L)
            self.u[index] = u_R * (self.x[index] - 1)

        elif self.init_type == "tophat":
            if boundary is None:
                self.boundary = "outflow"
            self.u[:] = 0.0
            self.u[np.logical_and(self.x >= 0.333,
                                       self.x <= 0.666)] = 1.0
        elif self.init_type == "sine":
            if boundary is None:
                self.boundary = "periodic"
            self.u[:] = 1.0
            index = np.logical_and(self.x >= 0.333,
                                   self.x <= 0.666)
            self.u[index] += \
                0.5 * np.sin(2.0 * np.pi * (self.x[index] - 0.333) / 0.333)

        elif self.init_type == "rarefaction":
            if boundary is None:
                self.boundary = "outflow"
            self.u[:] = 1.0
            self.u[self.x > 0.5] = 2.0

        elif self.init_type == "zero":
            if boundary is None:
                self.boundary = "periodic"
            self.u[:] = 0

        elif self.init_type == "flat":
            if boundary is None:
                self.boundary = "outflow"
            if 'a' in params:
                a = params['a']
            else:
                a = 2.0
            self.init_params['a'] = a
            self.u[:] = a

        elif self.init_type == "line":
            if boundary is None:
                self.boundary = "first"
            if 'a' in params:
                a = params['a']
            else:
                a = 2.0
            self.init_params['a'] = a
            if 'b' in params:
                b = params['b']
            else:
                b = -0.1
            self.init_params['b'] = b
            self.u = a + b * self.x
        else:
            raise Exception("Initial condition type \"" + str(self.init_type) + "\" not recognized.")

        self.init_params['boundary'] = self.boundary
        self.update_boundary()

    def update_boundary(self):
        if self.boundary == "first":
            left_d = self.u[self.ilo] - self.u[self.ilo+1]
            self.u[0:self.ilo] = (self.u[self.ilo] + (1 + np.arange(self.ng))*left_d)[::-1]
            right_d = self.u[self.ihi] - self.u[self.ihi-1]
            self.u[self.ihi+1:] = (self.u[self.ihi] + (1 + np.arange(self.ng))*right_d)
        else:
            super().update_boundary()

class Grid2d(GridBase):
    def __init__(self, num_cells, num_ghosts, min_value, max_value, boundary=None,
            init_type="gaussian", deterministic_init=False):
        super().__init__(num_cells, num_ghosts, min_value, max_value, boundary)

        # _init_type and _boundary do not change, init_type and boundary may
        # change if init_type is scheduled or sampled.
        self._init_type = init_type
        self.init_type = init_type
        self._boundary = boundary
        self.boundary = boundary
        self.deterministic_init = deterministic_init
        self._init_schedule_index = 0
        self._init_schedule = ["gaussian"]
        self._init_sample_types = self._init_schedule
        self._init_sample_probs = [1/len(self._init_sample_types)]*len(self._init_sample_types)

        self.init_params = {}


    def reset(self, params={}):

        if 'init_type' in params:
            self.init_type = params['init_type']
        elif self._init_type == "schedule":
            self.init_type = self._init_schedule[self._init_schedule_index]
            self._init_schedule_index = (self._init_schedule_index + 1) % len(self._init_schedule)
        elif self._init_type == "sample":
            self.init_type = np.random.choice(self._init_sample_types, p=self._init_sample_probs)

        if 'boundary' in params:
            self.boundary = params['boundary']
        else:
            self.boundary = self._boundary

        new_params = {'init_type': self.init_type}

        if self.init_type == "custom" or type(self.init_type) is not str:
            assert callable(self._init_type), "Custom init must have function provided as init type."
            assert boundary is not None, "Cannot use default boundary with custom init type."
            new_values, custom_params = self._init_type(params)
            new_params.update(custom_params)
            self.space[self.real_slice] = new_values

        elif self.init_type == "gaussian":
            if self.boundary is None:
                self.boundary = "outflow"

            if 'a' in params:
                a = params['a']
            else:
                a = 0.0
            new_params['a'] = a
            if 'b' in params:
                b = params['b']
            else:
                b = 1.0
            new_params['b'] = b
            if 'c' in params:
                c = params['c']
            else:
                c = tuple((max_val - min_val)/2 for max_val, min_val in
                        zip(self.max_value, self.min_value))
            new_params['c'] = c
            if 'sigma' in params:
                sigma = params['sigma']
            else:
                sigma = tuple(0.091 for _ in self.num_cells)
            new_params['sigma'] = sigma
            self.space = a + b*np.exp(-(
                (self.x[:, None] - c[0])**2/(2.0*sigma[0]**2)
                + (self.y[None, :] - c[1])**2/(2.0*sigma[1]**2)))

        new_params['boundary'] = self.boundary
        self.init_params = new_params
        
        self.update_boundary()
