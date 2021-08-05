# Originally adapted from code for
# Introduction to Computational Astrophysical Hydrodynamics
# by M. Zingale (2013-03-26).


import numpy as np

from envs.grid import GridBase

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

    def __init__(self, nx, ng, xmin=0.0, xmax=1.0,
                 init_type="sine", boundary=None, deterministic_init=False):
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
        super().__init__(nx, ng, xmin, xmax, vec_len, boundary)

        # _init_type and _boundary do not change, init_type and boundary may
        # change if init_type is scheduled or sampled.
        self._init_type = init_type
        self.init_type = init_type
        self._boundary = self.boundary
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


class EulerGrid1d(GridBase):
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

    def __init__(self, nx, ng, xmin=0.0, xmax=1.0, eos_gamma=1.4, init_type="double_rarefaction", boundary=None,
                 deterministic_init=False):
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
        super().__init__(nx, ng, xmin, xmax, vec_len, boundary)

        # _init_type and _boundary do not change, init_type and boundary may
        # change if init_type is scheduled or sampled.
        self.init_type = init_type
        self._boundary = self.boundary
        self.deterministic_init = deterministic_init

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

        if 'boundary' in params:
            boundary = params['boundary']
        else:
            boundary = self._boundary
        self.boundary = boundary

        self.init_params = {'init_type': self.init_type}

        if self.init_type == "sod":
            if boundary is None:
                self.boundary = "outflow"
            rho_l = 1
            rho_r = 1 / 8
            v_l = 0
            v_r = 0
            p_l = 1
            p_r = 1 / 10
            S_l = rho_l * v_l
            S_r = rho_r * v_r
            e_l = p_l / rho_l / (self.eos_gamma - 1)
            e_r = p_r / rho_r / (self.eos_gamma - 1)
            E_l = rho_l * (e_l + v_l ** 2 / 2)
            E_r = rho_r * (e_r + v_r ** 2 / 2)
            self.u[0] = np.where(self.x < 0, rho_l * np.ones_like(self.x), rho_r * np.ones_like(self.x))
            self.u[1] = np.where(self.x < 0, S_l * np.ones_like(self.x), S_r * np.ones_like(self.x))
            self.u[2] = np.where(self.x < 0, E_l * np.ones_like(self.x), E_r * np.ones_like(self.x))

        elif self.init_type == "double_rarefaction":
            if boundary is None:
                self.boundary = "outflow"
            rho_l = 1
            rho_r = 1
            v_l = -2
            v_r = 2
            p_l = 0.4
            p_r = 0.4
            S_l = rho_l * v_l
            S_r = rho_r * v_r
            e_l = p_l / rho_l / (self.eos_gamma - 1)
            e_r = p_r / rho_r / (self.eos_gamma - 1)
            E_l = rho_l * (e_l + v_l ** 2 / 2)
            E_r = rho_r * (e_r + v_r ** 2 / 2)
            self.u[0] = np.where(self.x < 0,  rho_l * np.ones_like(self.x), rho_r * np.ones_like(self.x))
            self.u[1] = np.where(self.x < 0, S_l * np.ones_like(self.x), S_r * np.ones_like(self.x))
            self.u[2] = np.where(self.x < 0, E_l * np.ones_like(self.x), E_r * np.ones_like(self.x))

        elif self.init_type == "slow_shock":
            if boundary is None:
                self.boundary = "outflow"
            rho_l = 5.6698
            rho_r = 1.0
            v_l = -1.4701
            v_r = -10.5
            p_l = 100.0
            p_r = 1.0
            S_l = rho_l * v_l
            S_r = rho_r * v_r
            e_l = p_l / rho_l / (self.eos_gamma - 1)
            e_r = p_r / rho_r / (self.eos_gamma - 1)
            E_l = rho_l * (e_l + v_l ** 2 / 2)
            E_r = rho_r * (e_r + v_r ** 2 / 2)
            self.u[0] = np.where(self.x < 0, rho_l * np.ones_like(self.x), rho_r * np.ones_like(self.x))
            self.u[1] = np.where(self.x < 0, S_l * np.ones_like(self.x), S_r * np.ones_like(self.x))
            self.u[2] = np.where(self.x < 0, E_l * np.ones_like(self.x), E_r * np.ones_like(self.x))
        
        elif type == "advection":
            if boundary is None:
                self.boundary = "outflow"
            x = self.x
            rho_0 = 1e-3
            rho_1 = 1
            sigma = 0.1
            rho = rho_0 * np.ones_like(x)
            rho += (rho_1 - rho_0) * np.exp(-(x - 0.5) ** 2 / sigma ** 2)
            v = np.ones_like(x)
            p = 1e-6 * np.ones_like(x)
            S = rho * v
            e = p / rho / (self.eos_gamma - 1)
            E = rho * (e + v ** 2 / 2)
            self.u[0, :] = rho[:]
            self.u[1, :] = S[:]
            self.u[2, :] = E[:]

        else:
            raise Exception("Initial condition type \"" + str(self.init_type) + "\" not recognized.")

        self.init_params['boundary'] = self.boundary
        self.update_boundary()

    def update_boundary(self):
        if self.boundary == "first":
            left_d = self.u[self.ilo] - self.u[self.ilo + 1]
            self.u[0:self.ilo] = (self.u[self.ilo] + (1 + np.arange(self.ng)) * left_d)[::-1]
            right_d = self.u[self.ihi] - self.u[self.ihi - 1]
            self.u[self.ihi + 1:] = (self.u[self.ihi] + (1 + np.arange(self.ng)) * right_d)
        else:
            super().update_boundary()
