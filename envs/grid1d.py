# Originally adapted from code for
# Introduction to Computational Astrophysical Hydrodynamics
# by M. Zingale (2013-03-26).


import numpy as np

from envs.grid import GridBase

class Burgers1DGrid(GridBase):
    """
    1-dimensional grid of values for use in modelling differential equations.
    
    Fields intended for external use:
    grid.u - Current values in the grid, including ghost cells.
    Burgers1DGrid.update() updates the ghost cells automatically. If updating grid.u directly,
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

    DEFAULT_SCHEDULE = ["smooth_sine", "rarefaction", "accelshock"]

    def __init__(self, nx, ng, xmin=0.0, xmax=1.0,
                 init_type="smooth_sine", boundary=None,
                 schedule=None, deterministic_init=False, dtype=np.float64):
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

        vec_len = 1  # didn't put into init as a parameter because it should be fixed, also didn't put in
        super().__init__(nx, ng, xmin, xmax, vec_len, boundary, dtype=dtype)

        if init_type is None:
            init_type = "smooth_sine"

        # _init_type and _boundary do not change, init_type and boundary may
        # change if init_type is scheduled or sampled.
        self._init_type = init_type
        self.init_type = init_type
        self._boundary = self.boundary
        self.deterministic_init = deterministic_init
        self._init_schedule_index = 0
        #self._init_schedule = ["smooth_rare", "smooth_sine", "random", "rarefaction", "accelshock"]
        if schedule is None:
            self._init_schedule = self.DEFAULT_SCHEDULE
        else:
            self._init_schedule = schedule
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

        if callable(self.init_type):
            assert boundary is not None, "Cannot use default boundary with custom init type."
            new_u, custom_params = self.init_type(params)
            self.init_params.update(custom_params)
            self.space[0, self.ng:-self.ng] = new_u

        elif self.init_type.startswith("smooth_sine"):
            if self.init_type == "smooth_sine_outflow":
                self.boundary = "outflow"
            elif boundary is None:
                self.boundary = "periodic"
            if 'A' in params:
                A = params['A']
            else:
                A = float(1.0 / (2.0 * np.pi * 0.1))
            self.init_params['A'] = A
            if self.init_type == "smooth_sine_shift":
                if 'phi' in params:
                    phi = params['phi']
                else:
                                    # Using a prime number of points feels like the right idea.
                    phi = 2 * np.pi * np.random.choice(np.linspace(0.0, 1.0, 23, endpoint=False))
                self.init_params['phi'] = phi
            else:
                phi = 0.0
            self.space[0] = A*np.sin(2 * np.pi * self.x + phi)

        elif self.init_type == "other_sine":
            # "random" initialized with k=4, b=1.0, a=2.5.
            if self.boundary is None:
                self.boundary = "periodic"
            k = 4
            self.init_params['k'] = 4
            b = 0.5
            self.init_params['b'] = 0.5
            a = 2.5
            self.init_params['a'] = 2.5
            self.space[0] = a + b * np.sin(k * np.pi * self.x)

        elif self.init_type == "gaussian":
            if boundary is None:
                self.boundary = "outflow"
            self.space[0] = 1.0 + np.exp(-60.0 * (self.x - 0.5) ** 2)


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
                if 'random' in params and params['random'] == 'cont':
                    b_size = np.random.uniform(0.2, 1.0)
                else:
                    b_size = np.random.choice(np.linspace(0.2, 1.0, 9))
                b_sign = np.random.randint(2) * 2 - 1
                b = b_sign * b_size
            self.init_params['b'] = b
            if 'a' in params:
                a = params['a']
            else:
                a = float(3.5 - np.abs(b))
            self.init_params['a'] = a
            self.space[0] = a + b * np.sin(k * np.pi * self.x)

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
                if 'random' in params and params['random'] == 'cont':
                    b_size = np.random.uniform(0.2, 1.0)
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
                if 'random' in params and params['random'] == 'cont':
                    a = np.random.uniform(-2.0, 2.0)
                else:
                    a = np.random.choice(np.linspace(-2.0, 2.0, 9))
            self.init_params['a'] = a
            self.space[0] = a + b * np.sin(k * np.pi * self.x)

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
            else:
                k = 60
            self.init_params['k'] = k
            if 'C' in params:
                C = params['C']
            else:
                C = 0.0
            self.init_params['C'] = C
            self.space[0] = C + A * np.tanh(k * (self.x - 0.5))

        elif self.init_type == "smooth_rare_periodic":
            self.boundary = "periodic"
            if 'A' in params:
                A = params['A']
            else:
                A = 1.0
            self.init_params['A'] = A
            if 'k' in params:
                k = params['k']
            else:
                k = 60
            self.init_params['k'] = k
            if 'C' in params:
                C = params['C']
            else:
                C = 0.0
            self.init_params['C'] = C
            self.space[0] = C + A * np.tanh(k * (self.x - 0.5))

        elif self.init_type == "smooth_rare_random":
            if boundary is None:
                self.boundary = "outflow"
            if 'A' in params:
                A = params['A']
            elif self.deterministic_init:
                A = 1.0
            else:
                if 'random' in params and params['random'] == 'cont':
                    A = np.random.uniform(0.25, 1.5)
                else:
                    A = np.random.choice(np.linspace(0.25, 1.5, 9))
            self.init_params['A'] = A
            if 'k' in params:
                k = params['k']
            elif self.deterministic_init:
                k = 60
            else:
                if 'random' in params and params['random'] == 'cont':
                    k = np.random.uniform(20, 100)
                else:
                    k = int(np.random.choice(np.linspace(20, 100, 9)))
            self.init_params['k'] = k
            if 'C' in params:
                C = params['C']
            elif self.deterministic_init:
                C = 0.0
            else:
                if 'random' in params and params['random'] == 'cont':
                    C = np.random.uniform(-1.0, 1.0)
                else:
                    C = np.random.choice(np.linspace(-1.0, 1.0, 9))
            self.init_params['C'] = C

            self.space[0] = C + A * np.tanh(k * (self.x - 0.5))

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
            self.space[0] = np.full_like(self.x, u_L)
            self.space[0, index] = u_R * (self.x[index] - 1)

        elif self.init_type == "accelshock_periodic":
            self.boundary = "periodic"

            offset = params['offset'] if 'offset' in params else 0.25
            self.init_params['offset'] = offset
            u_L = params['u_L'] if 'u_L' in params else 3.0
            self.init_params['u_L'] = u_L
            u_R = params['u_R'] if 'u_R' in params else 3.0
            self.init_params['u_R'] = u_R

            index = self.x > offset
            self.space[0] = np.full_like(self.x, u_L)
            self.space[0, index] = u_R * (self.x[index] - 1)

        elif self.init_type == "accelshock_random":
            if boundary is None:
                self.boundary = "outflow"

            if 'offset' in params:
                offset = params['offset']
            elif self.deterministic_init:
                offset = 0.25
            else:
                if 'random' in params and params['random'] == 'cont':
                    offset = np.random.uniform(0.0, 0.5)
                else:
                    offset = np.random.choice(np.linspace(0.0, 0.5, 9))
            self.init_params['offset'] = offset
            if 'u_L' in params:
                u_L = params['u_L']
            elif self.deterministic_init:
                u_L = 3.0
            else:
                if 'random' in params and params['random'] == 'cont':
                    u_L = np.random.uniform(0.5, 5.0)
                else:
                    u_L = np.random.choice(np.linspace(0.5, 5.0, 9))
            self.init_params['u_L'] = u_L
            if 'u_R' in params:
                u_R = params['u_R']
            elif self.deterministic_init:
                u_R = 3.0
            else:
                if 'random' in params and params['random'] == 'cont':
                    u_R = np.random.uniform(0.0, 5.0)
                else:
                    u_R = np.random.choice(np.linspace(0.0, 5.0, 9))
            self.init_params['u_R'] = u_R

            index = self.x > offset
            self.space[0] = np.full_like(self.x, u_L)
            self.space[0, index] = u_R * (self.x[index] - 1)

        elif self.init_type == "shock":
            if boundary is None:
                self.boundary = "outflow"
            offset = params['offset'] if 'offset' in params else 0.25
            self.init_params['offset'] = offset
            u_L = params['u_L'] if 'u_L' in params else 3.0
            self.init_params['u_L'] = u_L
            u_R = params['u_R'] if 'u_R' in params else 1.0
            self.init_params['u_R'] = u_R

            index = self.x > offset
            self.space[0] = np.full_like(self.x, u_L)
            self.space[0, index] = u_R

        elif self.init_type == "sawtooth":
            if boundary is None:
                self.boundary = "outflow"
            start = params['start'] if 'start' in params else 0.333
            self.init_params['start'] = start
            end = params['end'] if 'end' in params else 0.666
            self.init_params['end'] = end
            A = params['A'] if 'A' in params else 2.0

            self.space[0, :] = 0.0
            index = np.logical_and(self.x >= 0.333,
                                   self.x <= 0.666)
            self.space[0, np.logical_and(self.x >= start, self.x <= end)] = \
                    A * (self.x[index] - start) / (end - start)

        elif self.init_type == "tophat":
            if boundary is None:
                self.boundary = "outflow"
            x1 = params['x1'] if 'x1' in params else 1.0/3.0
            self.init_params['x1'] = x1
            x2 = params['x2'] if 'x2' in params else 2.0/3.0
            self.init_params['x2'] = x2
            u_L = params['u_L'] if 'u_L' in params else 0.0
            self.init_params['u_L'] = u_L
            u_M = params['u_M'] if 'u_M' in params else 1.0
            self.init_params['u_M'] = u_M
            u_R = params['u_R'] if 'u_R' in params else u_L
            self.init_params['u_R'] = u_R

            self.space[0] = np.full_like(self.x, u_L)
            right_index = self.x >= x2
            self.space[0, right_index] = u_R
            middle_index = np.logical_and(x1 < self.x, self.x < x2)
            self.space[0, middle_index] = u_M

        elif self.init_type == "sine":
            if boundary is None:
                self.boundary = "periodic"
            self.space[0, :] = 1.0
            index = np.logical_and(self.x >= 0.333,
                                   self.x <= 0.666)
            self.space[0, index] += \
                0.5 * np.sin(2.0 * np.pi * (self.x[index] - 0.333) / 0.333)

        elif self.init_type == "rarefaction":
            if boundary is None:
                self.boundary = "outflow"
            offset = params['offset'] if 'offset' in params else 0.5
            self.init_params['offset'] = offset
            u_L = params['u_L'] if 'u_L' in params else 1.0
            self.init_params['u_L'] = u_L
            u_R = params['u_R'] if 'u_R' in params else 2.0
            self.init_params['u_R'] = u_R

            index = self.x > offset
            self.space[0] = np.full_like(self.x, u_L)
            self.space[0, index] = u_R

        elif self.init_type == "zero":
            if boundary is None:
                self.boundary = "periodic"
            self.space[0, :] = 0

        elif self.init_type == "flat":
            if boundary is None:
                self.boundary = "outflow"
            if 'a' in params:
                a = params['a']
            else:
                a = 2.0
            self.init_params['a'] = a
            self.space[0, :] = a

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
            self.space[0] = a + b * self.x
        elif self.init_type == "para":
            if boundary is None:
                self.boundary = "outflow"
            if 'a' in params:
                a = params['a']
            else:
                a = 2.0
            self.init_params['a'] = a
            if 'b' in params:
                b = params['b']
            else:
                b = 0.0
            self.init_params['b'] = b
            if 'c' in params:
                c = params['c']
            else:
                c = -1.0
            self.init_params['c'] = c
            self.space[0] = a + b*self.x + c*self.x**2
        else:
            raise Exception("Initial condition type \"" + str(self.init_type) + "\" not recognized.")

        self.init_params['boundary'] = self.boundary
        self.update_boundary()

    def update_boundary(self):
        if self.boundary == "first":
            left_d = self.space[0, self.ilo] - self.space[0, self.ilo+1]
            self.space[0, 0:self.ilo] = (self.space[0, self.ilo] + (1 + np.arange(self.ng))*left_d)[::-1]
            right_d = self.space[0, self.ihi] - self.space[0, self.ihi-1]
            self.space[0, self.ihi+1:] = (self.space[0, self.ihi] + (1 + np.arange(self.ng))*right_d)
        elif self.boundary == "second":
            # This has weird behavior, probably don't use it.
            left_2d = (self.space[0, self.ilo] - 2*self.space[0, self.ilo+1]
                            + self.space[0, self.ilo+2])
            left_d = ((self.space[0, self.ilo] - self.space[0, self.ilo+1]) 
                    + (1 + np.arange(self.ng))*left_2d)
            self.space[0, :self.ilo] = (self.space[0, self.ilo] + left_d)[::-1]
            right_2d = (self.space[0, self.ihi] - 2*self.space[0, self.ihi-1]
                            + self.space[0, self.ihi-2])
            right_d = ((self.space[0, self.ihi] - self.space[0, self.ihi-1])
                    + (1 + np.arange(self.ng))*right_2d)
            self.space[0, self.ihi+1:] = (self.space[0, self.ihi] + right_d)
        else:
            super().update_boundary()


class Euler1DGrid(GridBase):
    """
    1-dimensional grid of values for use in modelling differential equations.

    Fields intended for external use:
    grid.u - Current values in the grid, including ghost cells.
    Burgers1DGrid.update() updates the ghost cells automatically. If updating grid.u directly,
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

    DEFAULT_SCHEDULE = ["sod", "shock_tube"]  # sod2 leads to NaNs during training

    def __init__(self, nx, ng, xmin=0.0, xmax=1.0, eos_gamma=1.4, init_type="double_rarefaction",
                 boundary=None, schedule=None,
                 deterministic_init=False, dtype=np.float64):
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

        vec_len = 3  # didn't put into init as a parameter because it should be fixed
        super().__init__(nx, ng, xmin, xmax, vec_len, boundary, dtype=dtype)

        if init_type is None:
            init_type = "double_rarefaction"

        # _init_type and _boundary do not change, init_type and boundary may
        # change if init_type is scheduled or sampled.
        self._init_type = init_type
        self.init_type = init_type
        self._boundary = self.boundary
        self.deterministic_init = deterministic_init
        self._init_schedule_index = 0
        if schedule is None:
            self._init_schedule = self.DEFAULT_SCHEDULE
        else:
            self._init_schedule = schedule
        self._init_sample_types = self._init_schedule
        self._init_sample_probs = [1/len(self._init_sample_types)]*len(self._init_sample_types)


        # 0 and len-1 are indexes for values beyond the boundary,
        # so create ilo and ihi as indexes to real values
        self.ilo = ng
        self.ihi = ng + nx - 1

        self.eos_gamma = eos_gamma  # Gamma law EOS

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

        if self.init_type == "sod":
            if boundary is None:
                self.boundary = "outflow"
            if 'eos_gamma' in self.init_params:
                eos_gamma = self.init_params['eos_gamma']
            else:
                eos_gamma = 1.4  # Gamma law EOS
            rho_l = 1
            rho_r = 1 / 8
            v_l = 0
            v_r = 0
            p_l = 1
            p_r = 1 / 10
            S_l = rho_l * v_l
            S_r = rho_r * v_r
            e_l = p_l / rho_l / (eos_gamma - 1)
            e_r = p_r / rho_r / (eos_gamma - 1)
            E_l = rho_l * (e_l + v_l ** 2 / 2)
            E_r = rho_r * (e_r + v_r ** 2 / 2)
            self.space[0] = np.where(self.x < 0, rho_l * np.ones_like(self.x), rho_r * np.ones_like(self.x))
            self.space[1] = np.where(self.x < 0, S_l * np.ones_like(self.x), S_r * np.ones_like(self.x))
            self.space[2] = np.where(self.x < 0, E_l * np.ones_like(self.x), E_r * np.ones_like(self.x))

        elif self.init_type == "double_rarefaction":
            if boundary is None:
                self.boundary = "outflow"
            if 'eos_gamma' in self.init_params:
                eos_gamma = self.init_params['eos_gamma']
            else:
                eos_gamma = 1.4  # Gamma law EOS
            rho_l = 1
            rho_r = 1
            v_l = -2
            v_r = 2
            p_l = 0.4
            p_r = 0.4
            S_l = rho_l * v_l
            S_r = rho_r * v_r
            e_l = p_l / rho_l / (eos_gamma - 1)
            e_r = p_r / rho_r / (eos_gamma - 1)
            E_l = rho_l * (e_l + v_l ** 2 / 2)
            E_r = rho_r * (e_r + v_r ** 2 / 2)
            self.space[0] = np.where(self.x < 0,  rho_l * np.ones_like(self.x), rho_r * np.ones_like(self.x))
            self.space[1] = np.where(self.x < 0, S_l * np.ones_like(self.x), S_r * np.ones_like(self.x))
            self.space[2] = np.where(self.x < 0, E_l * np.ones_like(self.x), E_r * np.ones_like(self.x))

        elif self.init_type == "slow_shock":
            if boundary is None:
                self.boundary = "outflow"
            if 'eos_gamma' in self.init_params:
                eos_gamma = self.init_params['eos_gamma']
            else:
                eos_gamma = 1.4  # Gamma law EOS
            rho_l = 5.6698
            rho_r = 1.0
            v_l = -1.4701
            v_r = -10.5
            p_l = 100.0
            p_r = 1.0
            S_l = rho_l * v_l
            S_r = rho_r * v_r
            e_l = p_l / rho_l / (eos_gamma - 1)
            e_r = p_r / rho_r / (eos_gamma - 1)
            E_l = rho_l * (e_l + v_l ** 2 / 2)
            E_r = rho_r * (e_r + v_r ** 2 / 2)
            self.space[0] = np.where(self.x < 0, rho_l * np.ones_like(self.x), rho_r * np.ones_like(self.x))
            self.space[1] = np.where(self.x < 0, S_l * np.ones_like(self.x), S_r * np.ones_like(self.x))
            self.space[2] = np.where(self.x < 0, E_l * np.ones_like(self.x), E_r * np.ones_like(self.x))
        
        elif type == "advection":
            if boundary is None:
                self.boundary = "outflow"
            if 'eos_gamma' in self.init_params:
                eos_gamma = self.init_params['eos_gamma']
            else:
                eos_gamma = 1.4  # Gamma law EOS
            x = self.x
            rho_0 = 1e-3
            rho_1 = 1
            sigma = 0.1
            rho = rho_0 * np.ones_like(x)
            rho += (rho_1 - rho_0) * np.exp(-(x - 0.5) ** 2 / sigma ** 2)
            v = np.ones_like(x)
            p = 1e-6 * np.ones_like(x)
            S = rho * v
            e = p / rho / (eos_gamma - 1)
            E = rho * (e + v ** 2 / 2)
            self.space[0, :] = rho[:]
            self.space[1, :] = S[:]
            self.space[2, :] = E[:]

        elif self.init_type == "sod2":
            if boundary is None:
                self.boundary = "outflow"
            if 'eos_gamma' in self.init_params:
                eos_gamma = self.init_params['eos_gamma']
            else:
                eos_gamma = 1.4  # Gamma law EOS
            rho_l = 1
            rho_r = 0.01
            v_l = 0
            v_r = 0
            p_l = 1
            p_r = 0.01
            S_l = rho_l * v_l
            S_r = rho_r * v_r
            e_l = p_l / rho_l / (eos_gamma - 1)
            e_r = p_r / rho_r / (eos_gamma - 1)
            E_l = rho_l * (e_l + v_l ** 2 / 2)
            E_r = rho_r * (e_r + v_r ** 2 / 2)
            self.space[0] = np.where(self.x < 0, rho_l * np.ones_like(self.x), rho_r * np.ones_like(self.x))
            self.space[1] = np.where(self.x < 0, S_l * np.ones_like(self.x), S_r * np.ones_like(self.x))
            self.space[2] = np.where(self.x < 0, E_l * np.ones_like(self.x), E_r * np.ones_like(self.x))

        elif self.init_type == "sonic_rarefaction":
            if boundary is None:
                self.boundary = "outflow"
            if 'eos_gamma' in self.init_params:
                eos_gamma = self.init_params['eos_gamma']
            else:
                eos_gamma = 1.4  # Gamma law EOS
            rho_l = 3.857
            rho_r = 1
            v_l = 0.92
            v_r = 3.55
            p_l = 10.333
            p_r = 1
            S_l = rho_l * v_l
            S_r = rho_r * v_r
            e_l = p_l / rho_l / (eos_gamma - 1)
            e_r = p_r / rho_r / (eos_gamma - 1)
            E_l = rho_l * (e_l + v_l ** 2 / 2)
            E_r = rho_r * (e_r + v_r ** 2 / 2)
            self.space[0] = np.where(self.x < 0, rho_l * np.ones_like(self.x), rho_r * np.ones_like(self.x))
            self.space[1] = np.where(self.x < 0, S_l * np.ones_like(self.x), S_r * np.ones_like(self.x))
            self.space[2] = np.where(self.x < 0, E_l * np.ones_like(self.x), E_r * np.ones_like(self.x))

        elif self.init_type == "slow_moving_shock":
            if boundary is None:
                self.boundary = "outflow"
            if 'eos_gamma' in self.init_params:
                eos_gamma = self.init_params['eos_gamma']
            else:
                eos_gamma = 1.4  # Gamma law EOS
            rho_l = 3.86
            rho_r = 1
            v_l = -0.81
            v_r = -3.44
            p_l = 10.33
            p_r = 1
            S_l = rho_l * v_l
            S_r = rho_r * v_r
            e_l = p_l / rho_l / (eos_gamma - 1)
            e_r = p_r / rho_r / (eos_gamma - 1)
            E_l = rho_l * (e_l + v_l ** 2 / 2)
            E_r = rho_r * (e_r + v_r ** 2 / 2)
            self.space[0] = np.where(self.x < 0, rho_l * np.ones_like(self.x), rho_r * np.ones_like(self.x))
            self.space[1] = np.where(self.x < 0, S_l * np.ones_like(self.x), S_r * np.ones_like(self.x))
            self.space[2] = np.where(self.x < 0, E_l * np.ones_like(self.x), E_r * np.ones_like(self.x))

        elif self.init_type == "shock_tube":
            if boundary is None:
                self.boundary = "outflow"
            if 'eos_gamma' in self.init_params:
                eos_gamma = self.init_params['eos_gamma']
            else:
                eos_gamma = 1.4  # Gamma law EOS
            rho_l = 0.445
            rho_r = 0.5
            v_l = 0.689
            v_r = 0
            p_l = 3.528
            p_r = 0.571
            S_l = rho_l * v_l
            S_r = rho_r * v_r
            e_l = p_l / rho_l / (eos_gamma - 1)
            e_r = p_r / rho_r / (eos_gamma - 1)
            E_l = rho_l * (e_l + v_l ** 2 / 2)
            E_r = rho_r * (e_r + v_r ** 2 / 2)
            self.space[0] = np.where(self.x < 0, rho_l * np.ones_like(self.x), rho_r * np.ones_like(self.x))
            self.space[1] = np.where(self.x < 0, S_l * np.ones_like(self.x), S_r * np.ones_like(self.x))
            self.space[2] = np.where(self.x < 0, E_l * np.ones_like(self.x), E_r * np.ones_like(self.x))

        else:
            raise Exception("Initial condition type \"" + str(self.init_type) + "\" not recognized.")

        self.init_params['boundary'] = self.boundary
        self.update_boundary()
