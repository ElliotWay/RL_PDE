import sys
import os
import numpy
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

import tensorflow as tf

tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

from envs import weno_coefficients
import riemann
from models.sac import SACBatch
from util.misc import create_stencil_indexes

import argparse
from argparse import Namespace
from envs import builder as env_builder
from models import get_model_arg_parser
from util.function_dict import numpy_fn
from util import metadata
from util.lookup import get_model_class, get_emi_class


class Grid1d(object):

    def __init__(self, nx, ng, xmin=0.0, xmax=1.0, bc="outflow"):

        self.ng = ng
        self.nx = nx

        self.xmin = xmin
        self.xmax = xmax

        self.bc = bc

        # python is zero-based.  Make easy intergers to know where the
        # real data lives
        self.ilo = ng
        self.ihi = ng + nx - 1

        # physical coords -- cell-centered, left and right edges
        self.dx = (xmax - xmin) / (nx)
        self.x = xmin + (numpy.arange(nx + 2 * ng) - ng + 0.5) * self.dx

        # storage for the solution
        self.q = numpy.zeros((3, (nx + 2 * ng)), dtype=numpy.float64)

    def scratch_array(self):
        """ return a scratch array dimensioned for our grid """
        return numpy.zeros((3, (self.nx + 2 * self.ng)), dtype=numpy.float64)

    def fill_BCs(self):
        """ fill all ghostcells as periodic """

        if self.bc == "periodic":

            # left boundary
            self.q[:, 0:self.ilo] = self.q[:, self.ihi - self.ng + 1:self.ihi + 1]

            # right boundary
            self.q[:, self.ihi + 1:] = self.q[:, self.ilo:self.ilo + self.ng]

        elif self.bc == "outflow":

            for n in range(self.ng):
                # left boundary
                self.q[:, n] = self.q[:, self.ilo]

                # right boundary
                self.q[:, self.ihi + 1 + n] = self.q[:, self.ihi]

        else:
            sys.exit("invalid BC")


# Used in step function to extract all the stencils.
def weno_stencils_batch(order, q_batch):
    """
    Take a batch of of stencils and approximate the target value in each sub-stencil.
    That is, for each stencil, approximate the target value multiple times using polynomial interpolation
    for different subsets of the stencil.
    This function relies on external coefficients for the polynomial interpolation.

    Parameters
    ----------
    order : int
      WENO sub-stencil width.
    q_batch : numpy array
      stencils for each location, shape is [2, grid_width+1, stencil_width].

    Returns
    -------
    Return a batch of stencils

    """

    a_mat = weno_coefficients.a_all[order]

    # These weights are "backwards" in the original formulation.
    # This is easier in the original formulation because we can add the k for our kth stencil to the index,
    # then subtract by a variable amount to get each value, but there's no need to do that here, and flipping
    # it back around makes the expression simpler.
    a_mat = np.flip(a_mat, axis=-1)

    sub_stencil_size = order
    num_sub_stencils = order
    # Adding a row vector and column vector gives us an "outer product" matrix where each row is a sub-stencil.
    sliding_window_indexes = np.arange(sub_stencil_size)[None, :] + np.arange(num_sub_stencils)[:, None]

    # [0,:,indexes] causes output to be transposed for some reason
    q_fp_stencil = np.sum(a_mat * q_batch[0][:, sliding_window_indexes], axis=-1)
    q_fm_stencil = np.sum(a_mat * q_batch[1][:, sliding_window_indexes], axis=-1)

    return np.array([q_fp_stencil, q_fm_stencil])

# Dheeraj - add
# weno function to work on single stencil

def weno_ii(order, q):
    """
    Do WENO reconstruction at a given location

    Parameters
    ----------

    order : int
        The stencil width
    q : numpy array
        Scalar data to reconstruct

    Returns
    -------

    qL : float
        Reconstructed data - boundary points are zero
    """
    C = weno_coefficients.C_all[order]
    a = weno_coefficients.a_all[order]
    sigma = weno_coefficients.sigma_all[order]

    qL = numpy.zeros(3)
    beta = numpy.zeros((order))
    w = numpy.zeros_like(beta)
    epsilon = 1e-16

    for nv in range(3):
        q_stencils = numpy.zeros(order)
        alpha = numpy.zeros(order)
        for k in range(order):
            for l in range(order):
                for m in range(l + 1):
                    beta[k] += sigma[k, l, m] * q[nv, order - 1 + k - l] * q[nv, order - 1 + k - m]
            alpha[k] = C[k] / (epsilon + beta[k] ** 2)
            for l in range(order):
                q_stencils[k] += a[k, l] * q[nv, order - 1 + k - l]
        w[:] = alpha / numpy.sum(alpha)
        qL[nv] = numpy.dot(w[:], q_stencils)

    return qL
#dheeraj -end

def weno(order, q):
    """
    Do WENO reconstruction

    Parameters
    ----------

    order : int
        The stencil width
    q : numpy array
        Scalar data to reconstruct

    Returns
    -------

    qL : numpy array
        Reconstructed data - boundary points are zero
    """
    C = weno_coefficients.C_all[order]
    a = weno_coefficients.a_all[order]
    sigma = weno_coefficients.sigma_all[order]

    qL = numpy.zeros_like(q)
    beta = numpy.zeros((order, q.shape[1]))
    w = numpy.zeros_like(beta)
    np = q.shape[1] - 2 * order
    epsilon = 1e-16
    for nv in range(3):
        for i in range(order, np + order):
            q_stencils = numpy.zeros(order)
            alpha = numpy.zeros(order)
            for k in range(order):
                for l in range(order):
                    for m in range(l + 1):
                        beta[k, i] += sigma[k, l, m] * q[nv, i + k - l] * q[nv, i + k - m]
                #                alpha[k] = C[k] / (epsilon + beta[k, i]**2)
                alpha[k] = C[k] / (epsilon + abs(beta[k, i]) ** order)
                for l in range(order):
                    q_stencils[k] += a[k, l] * q[nv, i + k - l]
            w[:, i] = alpha / numpy.sum(alpha)
            qL[nv, i] = numpy.dot(w[:, i], q_stencils)

    return qL


class WENOSimulation(object):

    def __init__(self, grid, C=0.5, weno_order=3, eos_gamma=1.4):
        self.grid = grid
        self.t = 0.0  # simulation time
        self.C = C  # CFL number
        self.weno_order = weno_order
        self.eos_gamma = eos_gamma  # Gamma law EOS

        self._state_axes = [None, None, None]

    def init_cond(self, type="sod"):
        if type == "sod":
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
            self.grid.q[0] = numpy.where(self.grid.x < 0,
                                         rho_l * numpy.ones_like(self.grid.x),
                                         rho_r * numpy.ones_like(self.grid.x))
            self.grid.q[1] = numpy.where(self.grid.x < 0,
                                         S_l * numpy.ones_like(self.grid.x),
                                         S_r * numpy.ones_like(self.grid.x))
            self.grid.q[2] = numpy.where(self.grid.x < 0,
                                         E_l * numpy.ones_like(self.grid.x),
                                         E_r * numpy.ones_like(self.grid.x))
        elif type == "advection":
            x = self.grid.x
            rho_0 = 1e-3
            rho_1 = 1
            sigma = 0.1
            rho = rho_0 * numpy.ones_like(x)
            rho += (rho_1 - rho_0) * numpy.exp(-(x - 0.5) ** 2 / sigma ** 2)
            v = numpy.ones_like(x)
            p = 1e-6 * numpy.ones_like(x)
            S = rho * v
            e = p / rho / (self.eos_gamma - 1)
            E = rho * (e + v ** 2 / 2)
            self.grid.q[0, :] = rho[:]
            self.grid.q[1, :] = S[:]
            self.grid.q[2, :] = E[:]
        elif type == "double_rarefaction":
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
            self.grid.q[0] = numpy.where(self.grid.x < 0,
                                         rho_l * numpy.ones_like(self.grid.x),
                                         rho_r * numpy.ones_like(self.grid.x))
            self.grid.q[1] = numpy.where(self.grid.x < 0,
                                         S_l * numpy.ones_like(self.grid.x),
                                         S_r * numpy.ones_like(self.grid.x))
            self.grid.q[2] = numpy.where(self.grid.x < 0,
                                         E_l * numpy.ones_like(self.grid.x),
                                         E_r * numpy.ones_like(self.grid.x))
        else:
            raise Exception("Invalid initial condition: \"{}\":".format(type))

    def max_lambda(self):
        rho = self.grid.q[0]
        v = self.grid.q[1] / rho
        p = (self.eos_gamma - 1) * (self.grid.q[2, :] - rho * v ** 2 / 2)
        cs = numpy.sqrt(self.eos_gamma * p / rho)
        return max(numpy.abs(v) + cs)

    def timestep(self):
        return self.C * self.grid.dx / self.max_lambda()

    def euler_flux(self, q):
        flux = numpy.zeros_like(q)
        rho = q[0, :]
        S = q[1, :]
        E = q[2, :]
        v = S / rho
        p = (self.eos_gamma - 1) * (E - rho * v ** 2 / 2)
        flux[0, :] = S
        flux[1, :] = S * v + p
        flux[2, :] = (E + p) * v
        return flux

    def rk_substep(self):

        g = self.grid
        g.fill_BCs()
        f = self.euler_flux(g.q)
        alpha = self.max_lambda()
        fp = (f + alpha * g.q) / 2
        fm = (f - alpha * g.q) / 2
        fpr = g.scratch_array()
        fml = g.scratch_array()
        flux = g.scratch_array()
        fpr[:, 1:] = weno(self.weno_order, fp[:, :-1])
        fml[:, -1::-1] = weno(self.weno_order, fm[:, -1::-1])
        flux[:, 1:-1] = fpr[:, 1:-1] + fml[:, 1:-1]
        rhs = g.scratch_array()
        rhs[:, 1:-1] = 1 / g.dx * (flux[:, 1:-1] - flux[:, 2:])
        return rhs

    def evecs(self, boundary_state):
        revecs = numpy.zeros((3, 3))
        levecs = numpy.zeros((3, 3))
        rho, S, E = boundary_state
        v = S / rho
        p = (self.eos_gamma - 1) * (E - rho * v ** 2 / 2)
        cs = numpy.sqrt(self.eos_gamma * p / rho)
        b1 = (self.eos_gamma - 1) / cs ** 2
        b2 = b1 * v ** 2 / 2
        revecs[0, 0] = 1
        revecs[0, 1] = v - cs
        revecs[0, 2] = (E + p) / rho - v * cs
        revecs[1, 0] = 1
        revecs[1, 1] = v
        revecs[1, 2] = v ** 2 / 2
        revecs[2, 0] = 1
        revecs[2, 1] = v + cs
        revecs[2, 2] = (E + p) / rho + v * cs
        levecs[0, 0] = (b2 + v / cs) / 2
        levecs[0, 1] = -(b1 * v + 1 / cs) / 2
        levecs[0, 2] = b1 / 2
        levecs[1, 0] = 1 - b2
        levecs[1, 1] = b1 * v
        levecs[1, 2] = -b1
        levecs[2, 0] = (b2 - v / cs) / 2
        levecs[2, 1] = -(b1 * v - 1 / cs) / 2
        levecs[2, 2] = b1 / 2
        return revecs, levecs

    def rk_substep_characteristic(self):
        """
        There is a major issue with the way that I've set up the weno
        function that means the code below requires the number of ghostzones
        to be weno_order+2, not just weno_order+1. This should not be needed,
        but I am too lazy to modify every weno routine to remove the extra,
        not required, point.

        The results aren't symmetric, so I'm not 100% convinced this is right.
        """

        g = self.grid
        g.fill_BCs()
        f = self.euler_flux(g.q)
        w_o = self.weno_order
        alpha = self.max_lambda()
        fp = (f + alpha * g.q) / 2
        fm = (f - alpha * g.q) / 2
        char_fm = g.scratch_array()
        char_fp = g.scratch_array()
        fpr = g.scratch_array()
        fml = g.scratch_array()
        flux = g.scratch_array()
        for i in range(g.ilo, g.ihi + 2):
            boundary_state = (g.q[:, i - 1] + g.q[:, i]) / 2
            revecs, levecs = self.evecs(boundary_state)
            for j in range(i - w_o - 1, i + w_o + 2):
                char_fm[:, j] = numpy.dot(fm[:, j], levecs)
                char_fp[:, j] = numpy.dot(fp[:, j], levecs)
            fpr[:, i - w_o:i + w_o + 2] = weno(self.weno_order,
                                               char_fp[:, i - w_o - 1:i + w_o + 1])
            fml[:, i + w_o + 1:i - w_o - 1:-1] = weno(self.weno_order,
                                                      char_fm[:, i + w_o + 1:i - w_o - 1:-1])
            flux[:, i] = numpy.dot(revecs, fpr[:, i] + fml[:, i])
        rhs = g.scratch_array()
        rhs[:, 1:-1] = 1 / g.dx * (flux[:, 1:-1] - flux[:, 2:])
        return rhs
#dheeraj - add

    def evecs_vectorized(self, boundary_state):
        g= self.grid
        ilo = g.ilo
        ihi = g.ihi
        real_length = g.nx + 2 * g.ng
        revecs = numpy.zeros((3, 3, real_length))
        levecs = numpy.zeros((3, 3, real_length))
        rho = boundary_state[0, :] #numpy.zeros((real_length))
        S = boundary_state[1, :] #numpy.zeros((real_length))
        E = boundary_state[2, :] #numpy.zeros((real_length))
        # rho, S, E = boundary_state[0, :]
        v = numpy.zeros((real_length))
        p = numpy.zeros((real_length))
        cs = numpy.zeros((real_length))
        indices = [_ for _ in range(ilo, ihi+2)]
        v[indices] = S[indices] / rho[indices]
        p[indices] = (self.eos_gamma - 1) * (E[indices] - rho[indices] * v[indices]**2 / 2)
        cs[indices] = numpy.sqrt(self.eos_gamma * p[indices] / rho[indices])
        b1 = (self.eos_gamma - 1) / cs[indices]**2
        b2 = b1 * v[indices]**2 / 2
        revecs[0, 0, indices] = 1
        revecs[0, 1, indices] = v[indices] - cs[indices]
        revecs[0, 2, indices] = (E[indices] + p[indices]) / rho[indices] - v[indices] * cs[indices]
        revecs[1, 0, indices] = 1
        revecs[1, 1, indices] = v[indices]
        revecs[1, 2, indices] = v[indices]**2 / 2
        revecs[2, 0, indices] = 1
        revecs[2, 1, indices] = v[indices] + cs[indices]
        revecs[2, 2, indices] = (E[indices] + p[indices]) / rho[indices] + v[indices] * cs[indices]
        levecs[0, 0, indices] = (b2 + v[indices] / cs[indices]) / 2
        levecs[0, 1, indices] = -(b1 * v[indices] + 1 / cs[indices]) / 2
        levecs[0, 2, indices] = b1 / 2
        levecs[1, 0, indices] = 1 - b2
        levecs[1, 1, indices] = b1 * v[indices]
        levecs[1, 2, indices] = -b1
        levecs[2, 0, indices] = (b2 - v[indices] / cs[indices]) / 2
        levecs[2, 1, indices] = -(b1 * v[indices] - 1 / cs[indices]) / 2
        levecs[2, 2, indices] = b1 / 2
        return revecs, levecs
    
    def rk_substep_characteristic_vector(self):
        """
        There is a major issue with the way that I've set up the weno
        function that means the code below requires the number of ghostzones
        to be weno_order+2, not just weno_order+1. This should not be needed,
        but I am too lazy to modify every weno routine to remove the extra,
        not required, point.

        The results aren't symmetric, so I'm not 100% convinced this is right.
        """

        g = self.grid
        g.fill_BCs()
        f = self.euler_flux(g.q)
        alpha = self.max_lambda()
        fp = (f + alpha * g.q) / 2
        fm = (f - alpha * g.q) / 2
        # compute the left fluxes
        fp_stencil_indexes = create_stencil_indexes(stencil_size=self.weno_order * 2 - 1,
                                                    num_stencils=g.nx + 1,
                                                    offset=g.ng - self.weno_order)

        fm_stencil_indexes = fp_stencil_indexes + 1

        fp_stencils = numpy.zeros((3, 2*self.weno_order-1, g.nx + 2*g.ng))
        indices = numpy.array([_ for _ in range(g.ilo, g.ihi + 2)])
        indices_minus_1 = indices - 1
        fp_stencils[:, :, indices] = numpy.transpose(numpy.array([fp[i][fp_stencil_indexes] for i in range(3)]), (0,2,1))
        fm_stencils = numpy.zeros((3, 2 * self.weno_order - 1, g.nx + 2 * g.ng))
        fm_stencils[:, :, indices] = numpy.transpose(numpy.array([fm[i][fm_stencil_indexes] for i in range(3)]), (0,2,1))

        boundary_state = numpy.zeros((3, g.nx + 2*g.ng))
        boundary_state[:, indices] = (g.q[:, indices_minus_1] + g.q[:, indices]) * 0.5
        revecs, levecs = self.evecs_vectorized(boundary_state)
        # first index - primary variable vector (=3)
        # second index - stencil size
        # third index - grid
        char_fp = numpy.einsum('jil, jkl-> kil', fp_stencils, levecs)
        char_fm = numpy.einsum('jil, jkl-> kil', fm_stencils, levecs)

        fpr = g.scratch_array()
        fml = g.scratch_array()

        # perhaps use apply_on_axis to vectorize this?
        for i in range(g.ilo, g.ihi+2):
            fpr[:, i] = weno_ii(order, char_fp[:, :, i])
            fml[:, i] = weno_ii(order, numpy.flip(char_fm[:, :, i], 1))


        flux = numpy.einsum('jil, il-> jl', revecs, fpr+fml)
        rhs = g.scratch_array()
        rhs[:, 1:-1] = 1 / g.dx * (flux[:, 1:-1] - flux[:, 2:])  # left - right
        return rhs
#dheeraj - end


    def evolve(self, tmax, reconstruction='componentwise'):
        """ evolve the Euler equation using RK4 """
        self.t = 0.0
        g = self.grid

        stepper = self.rk_substep
        if reconstruction == 'characteristic':
            stepper = self.rk_substep_characteristic

        step_count = 0
        self.save_plot("step0")

        # main evolution loop
        while self.t < tmax:

            # fill the boundary conditions
            g.fill_BCs()

            dt = self.timestep()

            if self.t + dt > tmax:
                dt = tmax - self.t

            # RK4
            # Store the data at the start of the step
            #            q_start = g.q.copy()
            #            k1 = dt * stepper()
            #            g.q = q_start + k1 / 2
            #            k2 = dt * stepper()
            #            g.q = q_start + k2 / 2
            #            k3 = dt * stepper()
            #            g.q = q_start + k3
            #            k4 = dt * stepper()
            #            g.q = q_start + (k1 + 2 * (k2 + k3) + k4) / 6

            # RK3: this is SSP
            # Store the data at the start of the step
            q_start = g.q.copy()
            q1 = q_start + dt * stepper()
            g.q[:, :] = q1[:, :]
            q2 = (3 * q_start + q1 + dt * stepper()) / 4
            g.q[:, :] = q2[:, :]
            g.q = (q_start + 2 * q2 + 2 * dt * stepper()) / 3

            step_count += 1
            if step_count % 5 == 0:
                self.save_plot(suffix="step{}".format(step_count))

            self.t += dt

    #            print("t=", self.t)

    def evolve_step(self, dt, reconstruction='componentwise'):
        stepper = self.rk_substep
        if reconstruction == 'characteristic':
            stepper = self.rk_substep_characteristic

        g = self.grid
        g.fill_BCs()

        q_start = g.q.copy()
        q1 = q_start + dt * stepper()
        g.q[:, :] = q1[:, :]
        q2 = (3 * q_start + q1 + dt * stepper()) / 4
        g.q[:, :] = q2[:, :]
        g.q = (q_start + 2 * q2 + 2 * dt * stepper()) / 3

    def rk_substep_agent(self, agent):
        g = self.grid
        g.fill_BCs()
        f = self.euler_flux(g.q)
        alpha = self.max_lambda()
        fp = (f + alpha * g.q) / 2
        fm = (f - alpha * g.q) / 2

        fp_indexes = create_stencil_indexes(stencil_size=(self.weno_order * 2 - 1),
                                            num_stencils=(g.nx + 1),
                                            offset=(g.ng - self.weno_order))
        fm_indexes = fp_indexes + 1
        fp_stencils = fp[:, fp_indexes]
        fm_stencils = np.flip(fm[:, fm_indexes], axis=-1)

        state = np.array([fp_stencils, fm_stencils])

        q_stencils = [weno_stencils_batch(self.weno_order, state[:, 0]),
                      weno_stencils_batch(self.weno_order, state[:, 1]),
                      weno_stencils_batch(self.weno_order, state[:, 2])]
        q_stencils = np.array(q_stencils)

        # The agent expects location to be first axis.
        # (fp, fm) X var X location X stencil -> var X location X (fp,fm) X stencil
        state = state.transpose((1, 2, 0, 3))

        weights_0, _ = agent.predict(state[0])
        weights_1, _ = agent.predict(state[1])
        weights_2, _ = agent.predict(state[2])
        weights = np.array([weights_0, weights_1, weights_2])

        weights = weights.transpose((0, 2, 1, 3))

        fml_and_fpr = np.sum(weights * q_stencils, axis=-1)

        flux = fml_and_fpr[:, 0] + fml_and_fpr[:, 1]

        rhs = g.scratch_array()
        rhs[:, g.ng:-g.ng] = (flux[:, :-1] - flux[:, 1:]) / g.dx

        return rhs

    def evolve_with_agent(self, agent, tmax, solution=None):
        self.t = 0.0
        g = self.grid

        step_count = 0
        self.save_plot("step000", other=solution)

        while self.t < tmax:
            print("t={}".format(self.t))
            dt = self.timestep()
            if self.t + dt > tmax:
                dt = tmax - self.t

            # Just Euler, skip RK for now.
            g.q += dt * self.rk_substep_agent(agent)

            if np.any(g.q[0] <= 0.0):
                print("Density dropped below 0!!!")
                # print(g.q[0])
                # self.save_plot(suffix="step{}_fail".format(step_count), other=solution)
                g.q[0] = np.clip(g.q[0], a_min=1e-6, a_max=None)
                # raise Exception()
            if np.any(g.q[2] <= 0.0):
                print("Energy dropped below 0!!!")
                # print(g.q[2])
                # self.save_plot(suffix="step{}_fail".format(step_count), other=solution)
                g.q[2] = np.clip(g.q[2], a_min=1e-6, a_max=None)
                # raise Exception()

            if solution is not None:
                solution.evolve_step(dt)

            step_count += 1
            if step_count % 1 == 0:
                self.save_plot(suffix="step{:03}".format(step_count), other=solution)

            self.t += dt

    def save_plot(self, suffix=None, other=None):
        fixed_axes = True
        no_x_border = True
        show_ghost = False

        fig, axes = plt.subplots(1, 3, sharex=True, sharey=True)

        full_x = self.grid.x
        real_x = full_x[self.grid.ng:-self.grid.ng]

        if other is not None:
            full_other = other.grid.q
            real_other = full_other[:, other.grid.ng:-other.grid.ng]
            other_color = "tab:blue"
            other_ghost_color = "#75bdf0"

        full_value = self.grid.q
        real_value = self.grid.q[:, self.grid.ng:-self.grid.ng]
        value_color = "tab:orange"
        value_ghost_color = "#ffad66"

        # The ghost arrays slice off one real point so the line connects to the real points.
        # Leave off labels for these lines so they don't show up in the legend.
        num_ghost_points = self.grid.ng + 1
        if show_ghost:
            ghost_x_left = full_x[:num_ghost_points]
            ghost_x_right = full_x[-num_ghost_points:]

            if other is not None:
                ghost_other_left = full_other[:, :num_ghost_points]
                ghost_other_right = full_other[:, -num_ghost_points:]
                for i in range(3):
                    axes[i].plot(ghost_x_left, ghost_other_left[i], ls='-', color=other_ghost_color)
                    axes[i].plot(ghost_x_right, ghost_other_right[i], ls='-', color=other_ghost_color)

            ghost_value_left = full_value[:, :num_ghost_points]
            ghost_value_right = full_value[:, -num_ghost_points:]

            for i in range(3):
                axes[i].plot(ghost_x_left, ghost_value_left[i], ls='-', color=value_ghost_color)
                axes[i].plot(ghost_x_right, ghost_value_right[i], ls='-', color=value_ghost_color)

        var_labels = ["\N{GREEK SMALL LETTER RHO}", "S", "E"]
        for i in range(3):
            axes[i].set_title(var_labels[i])
            if other is not None:
                axes[i].plot(real_x, real_other[i], ls='-', color=other_color, label="WENO")
            axes[i].plot(real_x, real_value[i], ls='-', color=value_color, label="agent")

            if no_x_border:
                axes[i].set_xmargin(0.0)

            if fixed_axes:
                if self._state_axes[i] is None:
                    self._state_axes[i] = (axes[i].get_xlim(), axes[i].get_ylim())
                else:
                    xlim, ylim = self._state_axes[i]
                    axes[i].set_xlim(xlim)
                    axes[i].set_ylim(ylim)

        title = "t = {:.3f}s".format(self.t)
        fig.suptitle(title)

        plt.legend()

        log_dir = LOG_DIR
        if suffix is None:
            suffix = ""
        filename = 'eulers' + suffix + '.png'
        filename = os.path.join(log_dir, filename)
        plt.savefig(filename)
        print('Saved plot to ' + filename + '.')

        plt.close(fig)


LOG_DIR = "test/weno_eulers/double_rarefaction"
# LOG_DIR = "test/weno_eulers/advection"
os.makedirs(LOG_DIR, exist_ok=True)


def setup_args():
    parser = argparse.ArgumentParser(
        description="Deploy an existing RL agent in an environment. Note that this script also takes various arguments not listed here.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--help-env', default=False, action='store_true',
                        help="Do not test and show the environment parameters not listed here.")
    parser.add_argument('--agent', '-a', type=str,
                        default="log/weno_burgers/full/210708_095448_newBC_order2/best_1_model_19420.zip",
                        # default="log/weno_burgers/full/210708_095536_newBC_order3/best_1_model_19970.zip",
                        help="Agent to test. Either a file or a string for a standard agent."
                             + " Parameters are loaded from 'meta.txt' in the same directory as the"
                             + " agent file, but can be overriden."
                             + " 'default' uses standard weno coefficients. 'none' forces no agent and"
                             + " only plots the true solution (ONLY IMPLEMENTED FOR EVOLUTION PLOTS).")
    parser.add_argument('--env', type=str, default="weno_burgers",
                        help="Name of the environment in which to deploy the agent.")
    parser.add_argument('--model', type=str, default='sac',
                        help="Type of model to be loaded. (Overrides the meta file.)")
    parser.add_argument('--emi', type=str, default='batch',
                        help="Environment-model interface. (Overrides the meta file.)")
    parser.add_argument('--obs-scale', '--obs_scale', type=str, default='z_score_last',
                        help="Adjustment function to observation. Compute Z score along the last"
                             + " dimension (the stencil) with 'z_score_last', the Z score along every"
                             + " dimension with 'z_score_all', or leave them the same with 'none'.")
    parser.add_argument('--action-scale', '--action_scale', type=str, default=None,
                        help="Adjustment function to action. Default depends on environment."
                             + " 'softmax' computes softmax, 'rescale_from_tanh' scales to [0,1] then"
                             + " divides by the sum of the weights, 'none' does nothing.")
    parser.add_argument('--log-dir', type=str, default=None,
                        help="Directory to place log file and other results. Default is test/env/agent/timestamp.")
    parser.add_argument('--ep-length', type=int, default=500,
                        help="Number of timesteps in an episode.")
    parser.add_argument('--seed', type=int, default=1,
                        help="Set random seed for reproducibility.")
    parser.add_argument('--plot-actions', '--plot_actions', default=False, action='store_true',
                        help="Plot the actions in addition to the state.")
    parser.add_argument('--animate', default=False, action='store_true',
                        help="Enable animation mode. Plot the state at every timestep, and keep the axes fixed across every plot.")
    parser.add_argument('--plot-error', '--plot_error', default=False, action='store_true',
                        help="Plot the error between the agent and the solution. Combines with evolution-plot.")
    parser.add_argument('--evolution-plot', '--evolution_plot', default=False, action='store_true',
                        help="Instead of usual rendering create 'evolution plot' which plots several states on the"
                             + " same plot in increasingly dark color.")
    parser.add_argument('--convergence-plot', '--convergence_plot', default=False, action='store_true',
                        help="Do several runs with different grid sizes to create a convergence plot."
                             " Overrides the --nx argument with 64, 128, 256, and 512, successively."
                             " Sets the --analytical flag.")
    parser.add_argument('--rk4', default=False, action='store_true',
                        help="Use RK4 steps instead of Euler steps. Only available for testing,"
                             + " since the reward during training doesn't make sense.")
    parser.add_argument('-y', default=False, action='store_true',
                        help="Choose yes for any questions, namely overwriting existing files. Useful for scripts.")
    parser.add_argument('-n', default=False, action='store_true',
                        help="Choose no for any questions, namely overwriting existing files. Useful for scripts. Overrides the -y option.")

    main_args, rest = parser.parse_known_args()

    env_arg_parser = env_builder.get_env_arg_parser()
    env_args, rest = env_arg_parser.parse_known_args(rest)
    env_builder.set_contingent_env_defaults(main_args, env_args)

    # run_test.py has model arguments that can be overidden, if desired,
    # but are intended to be loaded from a meta file.
    model_arg_parser = get_model_arg_parser()
    model_args, rest = model_arg_parser.parse_known_args(rest)

    internal_args = {}
    internal_args['total_episodes'] = 1000

    if main_args.env.startswith("weno"):
        mode = "weno"
    elif main_args.env.startswith("split_flux"):
        mode = "split_flux"
    elif main_args.env.startswith("flux"):
        mode = "flux"
    else:
        mode = "n/a"
    internal_args['mode'] = mode

    args = Namespace(**vars(main_args), **vars(env_args), **vars(model_args), **internal_args)

    return args


if __name__ == "__main__":

    xmin = -0.5
    xmax = 0.5
    nx = 128

    tmax = 0.1  # 0.1
    C = 0.5

    order = 2
    ng = order + 2

    init = "double_rarefaction"
    # init = "advection"

    left = riemann.State(p=0.4, u=-2.0, rho=1.0)
    right = riemann.State(p=0.4, u=2.0, rho=1.0)
    rp = riemann.RiemannProblem(left, right)
    rp.find_star_state()
    x_e, rho_e, v_e, p_e = rp.sample_solution(0.1, 1024)
    e_e = p_e / 0.4 / rho_e

    agent_grid = Grid1d(nx, ng, xmin, xmax, bc="outflow")
    agent_sim = WENOSimulation(agent_grid, C, order)
    agent_sim.init_cond(init)

    args = setup_args()
    metadata.load_to_namespace('log/weno_burgers/full/210708_095448_newBC_order2/meta.txt', args,
                               ignore_list=['log_dir', 'ep_length'])
    # metadata.load_to_namespace('log/weno_burgers/full/210708_095536_newBC_order3/meta.txt', args,
    #                            ignore_list=['log_dir', 'ep_length'])
    env = env_builder.build_env('weno_burgers', args, test=True)
    obs_adjust = numpy_fn(args.obs_scale)
    action_adjust = numpy_fn(args.action_scale)
    model_cls = get_model_class(args.model)
    emi_cls = get_emi_class(args.emi)
    emi = emi_cls(env, model_cls, args, obs_adjust=obs_adjust, action_adjust=action_adjust)
    emi.load_model(args.agent)
    agent = emi.get_policy()

    weno_grid = Grid1d(nx, ng, xmin, xmax, bc="outflow")
    weno_sim = WENOSimulation(weno_grid, C, order)
    weno_sim.init_cond(init)

    # agent_file = "possible_good_model.zip"
    # agent = SACBatch.load(agent_file)
    agent_sim.evolve_with_agent(agent, tmax, solution=weno_sim)
    agent_sim.save_plot("last", other=weno_sim)
    # weno_sim.evolve(tmax)
    # weno_sim.save_plot("last")

    g = agent_sim.grid
    x = g.x + 0.5
    rho = g.q[0, :]
    v = g.q[1, :] / g.q[0, :]
    e = (g.q[2, :] - rho * v ** 2 / 2) / rho
    p = (weno_sim.eos_gamma - 1) * (g.q[2, :] - rho * v ** 2 / 2)
    p2 = (agent_sim.eos_gamma - 1) * (g.q[2, :] - rho * v ** 2 / 2)
    fig, axes = plt.subplots(4, 1, sharex=True, figsize=(6, 10))
    axes[0].plot(x[g.ilo:g.ihi + 1], rho[g.ilo:g.ihi + 1], 'bo')
    axes[0].plot(x_e, rho_e, 'k--')
    axes[1].plot(x[g.ilo:g.ihi + 1], v[g.ilo:g.ihi + 1], 'bo')
    axes[1].plot(x_e, v_e, 'k--')
    axes[2].plot(x[g.ilo:g.ihi + 1], p[g.ilo:g.ihi + 1], 'bo')
    axes[2].plot(x[g.ilo:g.ihi + 1], p2[g.ilo:g.ihi + 1], 'r')
    axes[2].plot(x_e, p_e, 'k--')
    axes[2].legend(['weno', 'loaded agent', 'Godunov'])
    axes[3].plot(x[g.ilo:g.ihi + 1], e[g.ilo:g.ihi + 1], 'bo')
    axes[3].plot(x_e, e_e, 'k--')
    axes[0].set_ylabel(r"$\rho$")
    axes[1].set_ylabel(r"$u$")
    axes[2].set_ylabel(r"$p$")
    axes[3].set_xlabel(r"$x$")
    axes[3].set_ylabel(r"$e$")
    for ax in axes:
        ax.set_xlim(0, 1)
    axes[0].set_title(r"Double rarefaction, WENO, $r={}$".format(order))
    fig.tight_layout()
    plt.savefig(LOG_DIR + "weno-euler-rarefaction-r{}.png".format(order))
