import numpy as np

from envs.grid import GridBase

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
