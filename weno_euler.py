import sys
import numpy
import numpy as np
from matplotlib import pyplot

import tensorflow as tf

tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

import weno_coefficients
#import riemann
from models.sac import SACBatch
from util.misc import create_stencil_indexes

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
        self.ihi = ng+nx-1

        # physical coords -- cell-centered, left and right edges
        self.dx = (xmax - xmin)/(nx)
        self.x = xmin + (numpy.arange(nx+2*ng)-ng+0.5)*self.dx

        # storage for the solution
        self.q = numpy.zeros((3,(nx+2*ng)), dtype=numpy.float64)


    def scratch_array(self):
        """ return a scratch array dimensioned for our grid """
        return numpy.zeros((3, (self.nx+2*self.ng)), dtype=numpy.float64)


    def fill_BCs(self):
        """ fill all ghostcells as periodic """

        if self.bc == "periodic":

            # left boundary
            self.q[:, 0:self.ilo] = self.q[:, self.ihi-self.ng+1:self.ihi+1]

            # right boundary
            self.q[:, self.ihi+1:] = self.q[:, self.ilo:self.ilo+self.ng]

        elif self.bc == "outflow":

            for n in range(self.ng):
                # left boundary
                self.q[:, n] = self.q[:, self.ilo]
    
                # right boundary
                self.q[:, self.ihi+1+n] = self.q[:, self.ihi]

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
        for i in range(order, np+order):
            q_stencils = numpy.zeros(order)
            alpha = numpy.zeros(order)
            for k in range(order):
                for l in range(order):
                    for m in range(l+1):
                        beta[k, i] += sigma[k, l, m] * q[nv, i+k-l] * q[nv, i+k-m]
#                alpha[k] = C[k] / (epsilon + beta[k, i]**2)
                alpha[k] = C[k] / (epsilon + abs(beta[k, i])**order)
                for l in range(order):
                    q_stencils[k] += a[k, l] * q[nv, i+k-l]
            w[:, i] = alpha / numpy.sum(alpha)
            qL[nv, i] = numpy.dot(w[:, i], q_stencils)
    
    return qL


class WENOSimulation(object):
    
    def __init__(self, grid, C=0.5, weno_order=3, eos_gamma=1.4):
        self.grid = grid
        self.t = 0.0 # simulation time
        self.C = C   # CFL number
        self.weno_order = weno_order
        self.eos_gamma = eos_gamma # Gamma law EOS

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
            E_l = rho_l * (e_l + v_l**2 / 2)
            E_r = rho_r * (e_r + v_r**2 / 2)
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
            rho += (rho_1 - rho_0) * numpy.exp(-(x-0.5)**2/sigma**2)
            v = numpy.ones_like(x)
            p = 1e-6 * numpy.ones_like(x)
            S = rho * v
            e = p / rho / (self.eos_gamma - 1)
            E = rho * (e + v**2 / 2)
            self.grid.q[0, :] = rho[:]
            self.grid.q[1, :] = S[:]
            self.grid.q[2, :] = E[:]
        elif type == "double rarefaction":
            rho_l = 1
            rho_r = 1
            v_l =-2
            v_r = 2
            p_l = 0.4
            p_r = 0.4
            S_l = rho_l * v_l
            S_r = rho_r * v_r
            e_l = p_l / rho_l / (self.eos_gamma - 1)
            e_r = p_r / rho_r / (self.eos_gamma - 1)
            E_l = rho_l * (e_l + v_l**2 / 2)
            E_r = rho_r * (e_r + v_r**2 / 2)
            self.grid.q[0] = numpy.where(self.grid.x < 0,
                                         rho_l * numpy.ones_like(self.grid.x),
                                         rho_r * numpy.ones_like(self.grid.x))
            self.grid.q[1] = numpy.where(self.grid.x < 0,
                                         S_l * numpy.ones_like(self.grid.x),
                                         S_r * numpy.ones_like(self.grid.x))
            self.grid.q[2] = numpy.where(self.grid.x < 0,
                                         E_l * numpy.ones_like(self.grid.x),
                                         E_r * numpy.ones_like(self.grid.x))


    def max_lambda(self):
        rho = self.grid.q[0]
        v = self.grid.q[1] / rho
        p = (self.eos_gamma - 1) * (self.grid.q[2, :] - rho * v**2 / 2)
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
        p = (self.eos_gamma - 1) * (E - rho * v**2 / 2)
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
        rhs[:, 1:-1] = 1/g.dx * (flux[:, 1:-1] - flux[:, 2:])
        return rhs

    
    def evecs(self, boundary_state):
        revecs = numpy.zeros((3, 3))
        levecs = numpy.zeros((3, 3))
        rho, S, E = boundary_state
        v = S / rho
        p = (self.eos_gamma - 1) * (E - rho * v**2 / 2)
        cs = numpy.sqrt(self.eos_gamma * p / rho)
        b1 = (self.eos_gamma - 1) / cs**2
        b2 = b1 * v**2 / 2
        revecs[0, 0] = 1
        revecs[0, 1] = v - cs
        revecs[0, 2] = (E + p) / rho - v * cs
        revecs[1, 0] = 1
        revecs[1, 1] = v
        revecs[1, 2] = v**2 / 2
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
        for i in range(g.ilo, g.ihi+2):
            boundary_state = (g.q[:, i-1] + g.q[:, i]) / 2
            revecs, levecs = self.evecs(boundary_state)
            for j in range(i-w_o-1, i+w_o+2):
                char_fm[:, j] = numpy.dot(fm[:, j], levecs)
                char_fp[:, j] = numpy.dot(fp[:, j], levecs)
            fpr[:, i-w_o:i+w_o+2] = weno(self.weno_order,
                                           char_fp[:, i-w_o-1:i+w_o+1])
            fml[:, i+w_o+1:i-w_o-1:-1] = weno(self.weno_order,
                                               char_fm[:, i+w_o+1:i-w_o-1:-1])
            flux[:, i] = numpy.dot(revecs, fpr[:, i] + fml[:, i])
        rhs = g.scratch_array()
        rhs[:, 1:-1] = 1/g.dx * (flux[:, 1:-1] - flux[:, 2:])
        return rhs


    def evolve(self, tmax, reconstruction = 'componentwise'):
        """ evolve the Euler equation using RK4 """
        self.t = 0.0
        g = self.grid
        
        stepper = self.rk_substep
        if reconstruction == 'characteristic':
            stepper = self.rk_substep_characteristic

        # main evolution loop
        while self.t < tmax:

            # fill the boundary conditions
            g.fill_BCs()

            # get the timestep
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

            self.t += dt
#            print("t=", self.t)

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

        weights_0, _ = agent.predict(state[:, 0])
        weights_1, _ = agent.predict(state[:, 1])
        weights_2, _ = agent.predict(state[:, 2])
        weights = np.array([weights_0, weights_1, weights_2])

        fml_and_fpr = np.sum(weights * q_stencils, axis=-1)

        flux = fml_and_fpr[:,0] + fml_and_fpr[:,1]

        rhs = g.scratch_array()
        rhs[:, g.ng:-g.ng] = (flux[:, :-1] - flux[:, 1:]) / g.dx

        return rhs


    def evolve_with_agent(self, agent, tmax, reconstruction):
        self.t = 0.0
        g = self.grid

        while self.t < tmax:
            print("t={}".format(self.t))
            dt = self.timestep()
            if self.t + dt > tmax:
                dt = tmax - self.t

            # Just Euler, skip RK for now.
            g.q += dt * self.rk_substep_agent(agent)

            if np.any(g.q <= 0.0):
                print("Solution dropped below 0!!!")
                np.clip(g.q, a_min=1e-6)

            self.t += dt

    def save_plot(self, suffix=None):
        # TODO: change to render all 3 components of the state.
        fig = plt.figure()

        full_x = self.grid.x
        real_x = full_x[self.grid.ilo:self.grid.ihi + 1]

        #full_actual = self.grid.uactual
        #real_actual = full_actual[self.grid.ilo:self.grid.ihi + 1]
        #plt.plot(real_x, real_actual, ls='-', color='b', label="WENO")

        full_learned = self.grid.u
        real_learned = full_learned[self.grid.ilo:self.grid.ihi + 1]
        plt.plot(real_x, real_learned, ls='-', color='k', label="learned")

        show_ghost = True
        # The ghost arrays slice off one real point so the line connects to the real points.
        # Leave off labels for these lines so they don't show up in the legend.
        if show_ghost:
            ghost_x_left = full_x[:self.grid.ilo + 1]
            ghost_x_right = full_x[self.grid.ihi:]
            #ghost_actual_left = full_actual[:self.grid.ilo + 1]
            #ghost_actual_right = full_actual[self.grid.ihi:]
            #ghost_blue = "#80b0ff"
            #plt.plot(ghost_x_left, ghost_actual_left, ls='-', color=ghost_blue)
            #plt.plot(ghost_x_right, ghost_actual_right, ls='-', color=ghost_blue)
            ghost_learned_left = full_learned[:self.grid.ilo + 1]
            ghost_learned_right = full_learned[self.grid.ihi:]
            ghost_black = "#808080"  # ie grey
            plt.plot(ghost_x_left, ghost_learned_left, ls='-', color=ghost_black)
            plt.plot(ghost_x_right, ghost_learned_right, ls='-', color=ghost_black)

        plt.legend()

        ax = plt.gca()

        # Recalculate automatic axis scaling.
        ax.relim()
        ax.autoscale_view()

        ax.set_title("step=" + str(self.steps))

        log_dir = "test/euler/"#logger.get_dir()
        if suffix is None:
            step_precision = int(np.ceil(np.log(self.episode_length) / np.log(10)))
            suffix = ("_t{:0" + str(step_precision) + "}").format(self.steps)
        filename = 'burgers' + suffix + '.png'
        filename = os.path.join(log_dir, filename)
        plt.savefig(filename)
        print('Saved plot to ' + filename + '.')

        plt.close(fig)



if __name__ == "__main__":
 
    xmin = -0.5
    xmax = 0.5
    nx = 128
    
    tmax = 0.1
    C = 0.5

    order = 3
    ng = order+2

    g = Grid1d(nx, ng, xmin, xmax, bc="outflow")
    s = WENOSimulation(g, C, order)
    s.init_cond("advection")

    agent_file = "log/weno_burgers/sac/sine_init/720_3/model_final.zip"
    agent = SACBatch.load(agent_file)

    s.evolve_with_agent(agent, tmax, reconstruction="componentwise")

    s.save_plot()
