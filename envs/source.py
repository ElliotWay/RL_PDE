import numpy as np

from envs.grid import GridBase

class SourceBase(GridBase):
    """ SourceBase is the same as GridBase but indicates that subclasses are intended to be external sources. """
    pass

class RandomSource(SourceBase):

    def __init__(self, nx, ng, xmin, xmax, amplitude, k_min=1.0, k_max=3.0):
        super().__init__(nx=nx, ng=ng, xmin=xmin, xmax=xmax)

        ndoe = nx + 2 * ng
        a = 0.5 * amplitude * np.random.uniform(-1, 1, size=(ndoe, 1))
        #         omega = rs.uniform (-0.4, 0.4, size = (ndoe, 1))
        omega_max = 2.0 * np.pi * 10
        omega = np.random.uniform(-omega_max, omega_max, size=(ndoe, 1))
        k_values = np.arange(k_min, k_max + 1)
        k = np.random.choice(np.concatenate([-k_values, k_values]), size=(ndoe, 1))
        phi = np.random.uniform(0, 2 * np.pi, size=(ndoe, 1))
        spatial_phase = 2 * np.pi * k * 1 / (2 * np.pi)

        self.omega = omega
        self.spatial_phase = spatial_phase
        self.phi = phi
        self.a = a

        self.values = np.zeros(ndoe, dtype=np.float64)
    
    def update(self, dt, time):
        signals = np.sin(self.omega * time + self.spatial_phase + self.phi)
        self.values = (self.a * signals).flatten()

    def get_real(self):
        return self.values[self.ng:-self.ng]

    def get_full(self):
        return self.values

    def reset(self, **params):
        self.values[:] = 0.0


