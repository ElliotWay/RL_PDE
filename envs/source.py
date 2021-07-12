import numpy as np

from envs.grid import AbstractGrid

class SourceBase(AbstractGrid):
    """ SourceBase is the same as AbstractGrid but indicates that subclasses are intended to be external sources. """
    pass

class RandomSource(SourceBase):

    def __init__(self, grid, amplitude, k_min=3.0, k_max=6.0, num_mode=20, omega_max = 10):
        super().__init__(grid.nx, grid.ng, grid.xmin, grid.xmax)

        self.g = grid
        self.amplitude = amplitude
        self.k_min = k_min
        self.k_max = k_max
        self.num_mode = num_mode
        self.omega_max = omega_max

        self.reset()
    
    def update(self, dt, time):
        omega2 = self.omega * np.ones([1, self.g.nx])
        phi2 = self.phi * np.ones([1, self.g.nx])
        signals = self.a * np.sin(omega2 * time + self.spatial_phase + phi2)
        superposed_signals = np.sum(signals, axis=0)
        self.values[self.g.ilo:self.g.ihi+1] = superposed_signals.flatten()

    def get_real(self):
        return self.values[self.ng:-self.ng]

    def get_full(self):
        return self.values

    def reset(self):
        num_mode = self.num_mode
        a = 0.5 * self.amplitude * np.random.uniform(-1, 1, size=(num_mode, 1))
        #         omega = rs.uniform (-0.4, 0.4, size = (num_mode, 1))
        omega_max = self.omega_max
        omega = np.random.uniform(-omega_max, omega_max, size=(num_mode, 1))
        k_values = np.arange(self.k_min, self.k_max + 1)
        k = np.random.choice(np.concatenate([-k_values, k_values]), size=(num_mode, 1))
        phi = np.random.uniform(0, 2 * np.pi, size=(num_mode, 1))
        spatial_phase = 2 * np.pi * k * self.g.x[self.g.ilo:self.g.ihi+1] / (self.xmax-self.xmin)

        self.a = a
        self.omega = omega
        self.spatial_phase = spatial_phase
        self.phi = phi

        self.values = self.g.scratch_array()
