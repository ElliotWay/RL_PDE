# 2nd-order accurate finite-volume implementation of the inviscid Burger's
# equation with piecewise linear slope reconstruction
#
# We are solving u_t + u u_x = 0 with outflow boundary conditions
#
# M. Zingale (2013-03-26)

import sys

import matplotlib as mpl
import numpy as np

mpl.rcParams['mathtext.fontset'] = 'cm'
mpl.rcParams['mathtext.rm'] = 'serif'

mpl.rcParams['font.size'] = 12
mpl.rcParams['legend.fontsize'] = 'large'
mpl.rcParams['figure.titlesize'] = 'medium'


class Grid1d(object):

    def __init__(self, nx, ng, xmin=0.0, xmax=1.0, bc="outflow"):
        """
        Construct a 1D grid.
        
        This also represents grid cells "beyond" the boundary.

        Parameters
        ----------
        
        nx : int
            number of indexes in the discretized grid
        ng : int
            how many indexes beyond the boundary to include to represent
            boundary conditions, probably related to stencil size
        xmin : float
            coordinate of the beginning of the grid
        xmax : float
            coordinate of the end of the grid
        bc : float
            type of boundary condition
        """

        self.nx = nx
        self.ng = ng

        self.xmin = xmin
        self.xmax = xmax

        self.bc = bc

        # 0 and len-1 are indexes for values beyond the boundary,
        # so create ilo and ihi as indexes to real values
        self.ilo = ng
        self.ihi = ng + nx - 1

        # physical coords -- cell-centered, left and right edges
        self.dx = (xmax - xmin) / (nx)
        self.x = xmin + (np.arange(nx + 2 * ng) - ng + 0.5) * self.dx

        # storage for the solution
        self.u = np.zeros((nx + 2 * ng), dtype=np.float64)

        # storage for actual solution (for comparison to our approximated solution)
        self.uactual = np.zeros((nx + 2 * ng), dtype=np.float64)

    def scratch_array(self):
        """ return a scratch array dimensioned for our grid """
        return np.zeros((self.nx + 2 * self.ng), dtype=np.float64)

    def real_length(self):
        """ Return the number of indexes of real (non-ghost) points """
        return self.nx

    def full_length(self):
        """ Return the number of indexes of all points, including ghost points """
        return self.nx + 2 * self.ng

    def set_bc_type(self, new_bc):
        if not new_bc in ["periodic", "outflow"]:
            raise Exception("Invalid BC type.")
        else:
            self.bc = new_bc

    def fill_BCs(self):
        """ fill all ghostcells as periodic """

        if self.bc == "periodic":

            # left boundary
            self.u[0:self.ilo] = self.u[self.ihi - self.ng + 1:self.ihi + 1]
            self.uactual[0:self.ilo] = self.uactual[self.ihi - self.ng + 1:self.ihi + 1]

            # right boundary
            self.u[self.ihi + 1:] = self.u[self.ilo:self.ilo + self.ng]
            self.uactual[self.ihi + 1:] = self.uactual[self.ilo:self.ilo + self.ng]

        elif self.bc == "outflow":

            # left boundary
            self.u[0:self.ilo] = self.u[self.ilo]
            self.uactual[0:self.ilo] = self.uactual[self.ilo]

            # right boundary
            self.u[self.ihi + 1:] = self.u[self.ihi]
            self.uactual[self.ihi + 1:] = self.uactual[self.ihi]

        else:
            sys.exit("invalid BC")

    def norm(self, e):
        """ return the norm of quantity e which lives on the grid """
        if len(e) != 2 * self.ng + self.nx:
            return None

        return np.sqrt(self.dx * np.sum(e[self.ilo:self.ihi + 1] ** 2))


class Simulation(object):

    def __init__(self, grid, slope_type="godunov"):
        self.grid = grid
        self.t = 0.0
        self.slope_type = slope_type

    def init_cond(self, type="tophat"):

        if type == "tophat":
            self.grid.set_bc_type("outflow")
            self.grid.u[:] = 0.0
            self.grid.u[np.logical_and(self.grid.x >= 0.333,
                                       self.grid.x <= 0.666)] = 1.0

        elif type == "sine":
            self.grid.set_bc_type("periodic")
            self.grid.u[:] = 1.0

            index = np.logical_and(self.grid.x >= 0.333,
                                   self.grid.x <= 0.666)
            self.grid.u[index] += \
                0.5 * np.sin(2.0 * np.pi * (self.grid.x[index] - 0.333) / 0.333)

        elif type == "rarefaction":
            self.grid.set_bc_type("outflow")
            self.grid.u[:] = 1.0
            self.grid.u[self.grid.x > 0.5] = 2.0
        
        else:
            raise Exception("Initial condition type not recognized.")

    def timestep(self, C=None):
        if C is None:  # return a constant time step
            return 0.0005
        else:
            return C * self.grid.dx / max(abs(self.grid.u[self.grid.ilo:
                                                          self.grid.ihi + 1]))

    # new states
    def states_new(self, dt):
        """ compute the left and right interface states """

        g = self.grid
        # compute the piecewise linear slopes -- 2nd order MC limiter
        # we pick a range of cells that includes 1 ghost cell on either
        # side
        ib = g.ilo - 1
        ie = g.ihi + 1

        u = g.u

        slope = g.scratch_array()

        if self.slope_type == "godunov":
            slope[:] = 0.0
        elif self.slope_type == "centered":
            for i in range(g.ilo - 1, g.ihi + 2):
                slope[i] = 0.5 * (u[i + 1] - u[i - 1]) / g.dx

        # now the interface states.  Note that there are 1 more interfaces
        # than zones
        ul = g.scratch_array()
        ur = g.scratch_array()

        for i in range(g.ilo, g.ihi + 2):
            # left state on the current interface comes from zone i-1
            ul[i] = u[i - 1] + 0.5 * g.dx * (1.0 - u[i - 1] * dt / g.dx) * slope[i - 1]

            # right state on the current interface comes from zone i
            ur[i] = u[i] - 0.5 * g.dx * (1.0 + u[i] * dt / g.dx) * slope[i]

        return ul, ur

    def states(self, dt):
        """ compute the left and right interface states """

        g = self.grid
        # compute the piecewise linear slopes -- 2nd order MC limiter
        # we pick a range of cells that includes 1 ghost cell on either
        # side
        ib = g.ilo - 1
        ie = g.ihi + 1

        u = g.u

        # this is the MC limiter from van Leer (1977), as given in
        # LeVeque (2002).  Note that this is slightly different than
        # the expression from Colella (1990)

        dc = g.scratch_array()
        dl = g.scratch_array()
        dr = g.scratch_array()

        dc[ib:ie + 1] = 0.5 * (u[ib + 1:ie + 2] - u[ib - 1:ie])
        dl[ib:ie + 1] = u[ib + 1:ie + 2] - u[ib:ie + 1]
        dr[ib:ie + 1] = u[ib:ie + 1] - u[ib - 1:ie]

        # these where's do a minmod()
        d1 = 2.0 * np.where(np.fabs(dl) < np.fabs(dr), dl, dr)
        d2 = np.where(np.fabs(dc) < np.fabs(d1), dc, d1)
        ldeltau = np.where(dl * dr > 0.0, d2, 0.0)

        # now the interface states.  Note that there are 1 more interfaces
        # than zones
        ul = g.scratch_array()
        ur = g.scratch_array()

        # are these indices right?
        #
        #  --+-----------------+------------------+
        #     ^       i       ^ ^        i+1
        #     ur(i)     ul(i+1) ur(i+1)
        #
        ur[ib:ie + 2] = u[ib:ie + 2] - \
                        0.5 * (1.0 + u[ib:ie + 2] * dt / self.grid.dx) * ldeltau[ib:ie + 2]

        ul[ib + 1:ie + 2] = u[ib:ie + 1] + \
                            0.5 * (1.0 - u[ib:ie + 1] * dt / self.grid.dx) * ldeltau[ib:ie + 1]

        return ul, ur

    def riemann(self, ul, ur):
        """
        Riemann problem for Burgers' equation.
        """

        S = 0.5 * (ul + ur)
        ushock = np.where(S > 0.0, ul, ur)
        ushock = np.where(S == 0.0, 0.0, ushock)

        # rarefaction solution
        urare = np.where(ur <= 0.0, ur, 0.0)
        urare = np.where(ul >= 0.0, ul, urare)

        us = np.where(ul > ur, ushock, urare)

        return 0.5 * us * us

    def update(self, dt, flux):
        """ conservative update """

        g = self.grid

        unew = g.scratch_array()

        unew[g.ilo:g.ihi + 1] = g.u[g.ilo:g.ihi + 1] + \
                                dt / g.dx * (flux[g.ilo:g.ihi + 1] - flux[g.ilo + 1:g.ihi + 2])

        return unew

    def evolve(self, C, tmax):

        self.t = 0.0

        g = self.grid

        # main evolution loop
        while (self.t < tmax):

            # fill the boundary conditions
            g.fill_BCs()

            # get the timestep
            dt = self.timestep(C)

            if (self.t + dt > tmax):
                dt = tmax - self.t

            # get the interface states
            ul, ur = self.states_new(dt)

            # solve the Riemann problem at all interfaces
            flux = self.riemann(ul, ur)

            # do the conservative update
            unew = self.update(dt, flux)

            self.grid.u[:] = unew[:]

            self.t += dt
