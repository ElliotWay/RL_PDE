import re
import numpy as np

from envs.grid import GridBase
from envs.grid1d import Burgers1DGrid

class Burgers2DGrid(GridBase):
    
    # Not really a schedule.
    DEFAULT_SCHEDULE = ["gaussian"]

    def __init__(self, num_cells, num_ghosts, min_value, max_value, boundary=None,
            init_type="gaussian", schedule=None, deterministic_init=False, dtype=np.float64):
        vec_len = 1
        super().__init__(num_cells, num_ghosts, min_value, max_value, vec_len, boundary,
                dtype=dtype)

        if init_type is None:
            init_type = "gaussian"

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

        self.init_params = {}

    def reset(self, params={}):

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
            self.boundary = params['boundary']
        else:
            self.boundary = self._boundary

        new_params = {'init_type': self.init_type}

        if callable(self.init_type):
            assert boundary is not None, "Cannot use default boundary with custom init type."
            new_values, custom_params = self.init_type(params)
            new_params.update(custom_params)
            self.space[0, self.real_slice] = new_values

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
            self.space[0] = a + b*np.exp(-(
                (self.x[:, None] - c[0])**2/(2.0*sigma[0]**2)
                + (self.y[None, :] - c[1])**2/(2.0*sigma[1]**2)))
        # Jiang, Shu, Zhang, Example 7. (An alternative formulation of finite difference WENO
        # schemes with Lax-Wendroff time discretization for conservation laws)
        elif self.init_type == "jsz7":
            if self.boundary is None:
                self.boundary = "periodic"
            
            a = params['a'] if 'a' in params else 0.5
            new_params['a'] = a
            b = params['b'] if 'b' in params else 1.0
            new_params['b'] = b
            c = params['c'] if 'c' in params else (np.pi / 2.0,) * 2
            new_params['c'] = c

            self.space[0] = a + b * np.sin(c[0] * self.x[:, None] + c[1] * self.x[None, :])

        # Extension of 1D environment.
        elif self.init_type.startswith("1d"):
            one_d_type = None
            one_d_axis = None
            match = re.fullmatch("1d-([^-]*)-(x|y)", self.init_type)
            if match is not None:
                one_d_type = match[1]
                one_d_axis = match[2]
            else:
                match = re.fullmatch("1d-([^-]*)", self.init_type)
                if match is not None:
                    one_d_type = match[1]
                else:
                    if not self.init_type == "1d":
                        raise ValueError("Burgers2DGrid: Malformed 1d init type string"
                                + " \"{}\".".format(self.init_type)
                                + " Expecting strings like \"1d-sine-x\".")
            # The params dict overrides the sub-type passed in the full init_type.
            if 'type' in params:
                one_d_type = params['type']
            elif one_d_type is None:
                one_d_type = "smooth_sine"
            new_params['type'] = one_d_type
            if 'axis' in params:
                one_d_axis = params['axis']
            elif one_d_axis is None:
                one_d_axis = 'x'
            new_params['axis'] = one_d_axis

            #TODO: Handle explicit boundary conditions. Not sure how that works, but if the
            # boundary is ("periodic", "outflow"), that's what it should be, even if that doesn't
            # make sense.

            one_d_params = dict(params)
            if 'boundary' in one_d_params: # Always use the default boundary from Burgers1DGrid.
                del one_d_params['boundary']
            one_d_params['init_type'] = one_d_type

            # Keep track of the miscellaneous parameters.
            for param, value in one_d_params.items():
                if param not in new_params:
                    new_params[param] = value

            if one_d_axis == 'x':
                if not hasattr(self, "x_grid1d"):
                    self.x_grid1d = Burgers1DGrid(self.num_cells[0], self.num_ghosts[0],
                                                  self.min_value[0], self.max_value[0])
                self.x_grid1d.reset(params=one_d_params)
                x_bound = self.x_grid1d.boundary if type(self.x_grid1d.boundary) is str \
                            else self.x_grid1d.boundary[0]
                x_grid = self.x_grid1d.get_real()
                # The first 1 of the shape passed to np.tile is for the vector dimension.
                self.space[self.real_slice] = np.tile(x_grid[..., None], (1, 1, self.num_cells[1]))
                self.boundary = (x_bound, "outflow")
            elif one_d_axis == 'y':
                if not hasattr(self, "y_grid1d"):
                    self.y_grid1d = Burgers1DGrid(self.num_cells[1], self.num_ghosts[1],
                                                  self.min_value[1], self.max_value[1])
                self.y_grid1d.reset(params=one_d_params)
                y_bound = self.y_grid1d.boundary if type(self.y_grid1d.boundary) is str \
                            else self.y_grid1d.boundary[0]
                y_grid = self.y_grid1d.get_real()
                self.space[self.real_slice] = np.tile(y_grid, (1, self.num_cells[0], 1))
                self.boundary = ("outflow", y_bound)

        # Combine 2 1D initial conditions on each axis additively.
        elif self.init_type.startswith("combo"):
            x_type = None
            y_type = None
            match = re.fullmatch("combo-([^-]*)-([^-]*)", self.init_type)
            if match is not None:
                x_type = match[1]
                y_type = match[2]
            else:
                match = re.fullmatch("combo-([^-]*)", self.init_type)
                if match is not None:
                    x_type = match[1]
                    y_type = match[1]
                else:
                    if not self.init_type == "combo":
                        raise ValueError("Burgers2DGrid: Malformed combo init type string"
                                + " \"{}\".".format(self.init_type)
                                + " Expecting strings like \"combo-smooth_sine-gaussian\".")
                    else:
                        x_type = "smooth_sine"
                        y_type = "smooth_sine"
            # The params dict overrides the sub-types passed in the full init_type.
            if 'x-type' in params:
                x_type = params['x-type']
            new_params['x-type'] = x_type
            if 'y-type' in params:
                y_type = params['y-type']
            new_params['y-type'] = y_type

            # Always use the default boundary from Burgers1DGrid.
            if 'boundary' in params:
                del params['boundary']
 
            # Keep track of the miscellaneous parameters.
            for param, value in params.items():
                if param not in new_params:
                    new_params[param] = value

            x_params = {'init_type': x_type}
            y_params = {'init_type': y_type}
            for param, value in params.items():
                if param.startswith("x-") and param != "x-type":
                    x_params[param[2:]] = value
                elif param.startswith("y-") and param != "y-type":
                    y_params[param[2:]] = value

            if not hasattr(self, "x_grid1d"):
                self.x_grid1d = Burgers1DGrid(self.num_cells[0], self.num_ghosts[0],
                                              self.min_value[0], self.max_value[0])
            if not hasattr(self, "y_grid1d"):
                self.y_grid1d = Burgers1DGrid(self.num_cells[1], self.num_ghosts[1],
                                              self.min_value[1], self.max_value[1])

            self.x_grid1d.reset(params=x_params)
            x_bound = self.x_grid1d.boundary if type(self.x_grid1d.boundary) is str \
                        else self.x_grid1d.boundary[0]
            x_grid = self.x_grid1d.get_real()

            self.y_grid1d.reset(params=y_params)
            y_bound = self.y_grid1d.boundary if type(self.y_grid1d.boundary) is str \
                        else self.y_grid1d.boundary[0]
            y_grid = self.y_grid1d.get_real()

            self.space[self.real_slice] = x_grid[:, :, None] + y_grid[:, None, :]
            self.boundary = (x_bound, y_bound)

        else:
            raise Exception("Initial condition type \"" + str(self.init_type) + "\" not recognized.")

        new_params['boundary'] = self.boundary
        self.init_params = new_params
        
        self.update_boundary()
