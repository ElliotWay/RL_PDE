# 1D class adapted from code for Introduction to Computational Astrophysical Hydrodynamics
# Written by M. Zingale (2013-03-26).

import sys

import numpy as np

class GridBase:
    """
    Abstract base class for Grids.

    A Grid represents a quantity in a physical space that has been discretized
    into cells. A Grid contains real cells, representing the values within the
    physical space, and ghost cells, representing values just beyond the
    boundary of the physical space necessary for computing convolutions over
    the phyiscal space.

    A grid can be reset to a state specified by a standard parameter dict,
    and set using a list of real values (the ghost cells should be updated
    using some other means).
    """

    def __init__(self, nx, ng, xmin, xmax):
        self.xmin = xmin
        self.xmax = xmax
        self.nx = nx
        self.ng = ng

        self.dx = (xmax - xmin) / (nx)

        # Physical coordinates: cell-centered, left and right edges
        self.x = xmin + (np.arange(nx + 2 * ng) - ng + 0.5) * self.dx
        self.real_x = self.x[ng:-ng]

        # Physical coordinates: interfaces, left edges
        self.inter_x = xmin + (np.arange(nx + 2 * ng) - ng) * self.dx
    
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
        return np.zeros((self.nx + 2 * self.ng), dtype=np.float64)


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

        super().__init__(nx=nx, ng=ng, xmin=xmin, xmax=xmax)

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

        # storage for the solution
        self.u = np.zeros((nx + 2 * ng), dtype=np.float64)

    def get_real(self):
        """ Get the real (non-ghost) values in the grid. """
        return self.u[self.ng:-self.ng]
    def get_full(self):
        """ Get the full grid, including ghost cells. """
        return self.u
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

        if 'init_type' in params:
            self.init_type = params['init_type']
        elif self._init_type == "schedule":
            self.init_type = self._init_schedule[self._init_schedule_index]
            self._init_schedule_index = (self._init_schedule_index + 1) % len(self._init_schedule)
        elif self._init_type == "sample":
            self.init_type = np.random.choice(self._init_sample_types, p=self._init_sample_probs)

        boundary = self._boundary
        if 'boundary' in params:
            boundary = params['boundary']
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
            index = self.x > 0.25
            #self.u = np.full_like(self.x, 3)
            self.u[:] = 3
            self.u[index] = 3 * (self.x[index] - 1)

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
                b = -0.0001
            self.init_params['b'] = b
            self.u = a + b * self.x
        else:
            raise Exception("Initial condition type \"" + str(self.init_type) + "\" not recognized.")

        self.init_params['boundary'] = self.boundary
        self.update_boundary()

    def set(self, new_values):
        """
        Update the grid with new values.
        The ghost cells are also updated.

        Parameters
        ---------
        new_values : ndarray
            Array of new values, len(new_values) should be nx,
            ie it should not include ghost cells.
        """
        self.u[self.ng:-self.ng] = new_values
        self.update_boundary()

    def update_boundary(self):
        """
        Update the ghost cells based on the value of the Grid.boundary field.
        You should not need to call this - Grid.set calls this method.
        """
        # Periodic - copying from the opposite end, as if the space wraps around
        if self.boundary == "periodic":
            self.u[0:self.ilo] = self.u[self.ihi - self.ng + 1:self.ihi + 1]
            self.u[self.ihi + 1:] = self.u[self.ilo:self.ilo + self.ng]
        # Outflow - copy the edge values into the boundaries
        elif self.boundary == "outflow":
            self.u[0:self.ilo] = self.u[self.ilo]
            self.u[self.ihi + 1:] = self.u[self.ihi]
        # First - assume the first derivative stays constant based on the 2 edge values
        elif self.boundary == "first":
            left_d = self.u[self.ilo] - self.u[self.ilo+1]
            self.u[0:self.ilo] = (self.u[self.ilo] + (1 + np.arange(self.ng))*left_d)[::-1]
            right_d = self.u[self.ihi] - self.u[self.ihi-1]
            self.u[self.ihi+1:] = (self.u[self.ihi] + (1 + np.arange(self.ng))*right_d)
        else:
            raise Exception("Boundary type \"" + str(self.boundary) + "\" not recognized.")
