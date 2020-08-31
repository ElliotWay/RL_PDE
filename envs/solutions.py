import numpy as np
from scipy.optimize import fixed_point

import envs.weno_coefficients as weno_coefficients
from envs.grid import GridBase, Grid1d


class SolutionBase(GridBase):
    """ SolutionBase is the same as GridBase but indicates that subclasses are intended to be solutions. """
    def is_recording_actions(self):
        # Override this if you are indeed recording actions.
        return False

class PreciseWENOSolution(SolutionBase):
    #TODO: should also calculate precise WENO with smaller timesteps.

    def __init__(self, nx, ng, xmin, xmax,
                 precise_order, precise_scale, init_type, boundary, flux_function,
                 eps=0.0, source=None,
                 record_actions=None):
        super().__init__(nx=nx, ng=ng, xmin=xmin, xmax=xmax)

        assert (precise_scale % 2 == 1), "Precise scale must be odd for easier downsampling."

        self.precise_scale = precise_scale

        self.precise_nx = precise_scale * nx
        if precise_order + 1 > precise_scale * ng:
            self.precise_ng = precise_order + 1
            self.extra_ghosts = self.precise_ng - (precise_scale * ng)
        else:
            self.precise_ng = precise_scale * ng
            self.extra_ghosts = 0
        self.precise_grid = Grid1d(xmin=xmin, xmax=xmax, nx=self.precise_nx, ng=self.precise_ng,
                                   boundary=boundary, init_type=init_type)

        self.flux_function = flux_function
        self.eps = eps
        self.order = precise_order
        self.source = source

        self.record_action = record_actions
        if record_actions is not None:
            self.action_history = []

    def is_recording_actions(self):
        return (self.record_actions is not None)
    def set_record_actions(self, record_mode):
        self.record_actions = record_mode
        self.action_history = []
    def get_action_history(self):
        return self.action_history

    def weno_stencils(self, q):
        """
        Compute WENO stencils

        Parameters
        ----------
        q : np array
          Scalar data to reconstruct.

        Returns
        -------
        stencils

        """
        order = self.order
        a = weno_coefficients.a_all[order]
        num_points = len(q) - 2 * order
        q_stencils = np.zeros((order, len(q)))
        for i in range(order, num_points + order):
            for k in range(order):
                for l in range(order):
                    q_stencils[k, i] += a[k, l] * q[i + k - l]

        return q_stencils

    def weno_weights(self, q):
        """
        Compute WENO weights

        Parameters
        ----------
        q : np array
          Scalar data to reconstruct.

        Returns
        -------
        stencil weights

        """
        order = self.order
        C = weno_coefficients.C_all[order]
        sigma = weno_coefficients.sigma_all[order]

        beta = np.zeros((order, len(q)))
        w = np.zeros_like(beta)
        num_points = len(q) - 2 * order
        epsilon = 1e-16
        for i in range(order, num_points + order):
            alpha = np.zeros(order)
            for k in range(order):
                for l in range(order):
                    for m in range(l + 1):
                        beta[k, i] += sigma[k, l, m] * q[i + k - l] * q[i + k - m]
                alpha[k] = C[k] / (epsilon + beta[k, i] ** 2)
            w[:, i] = alpha / np.sum(alpha)

        return w

    def weno_new(self, q):
        """
        Compute WENO reconstruction

        Parameters
        ----------
        q : numpy array
          Scalar data to reconstruct.

        Returns
        -------
        qL: numpy array
          Reconstructed data.

        """

        weights = self.weno_weights(q)
        q_stencils = self.weno_stencils(q)
        qL = np.zeros_like(q)
        num_points = len(q) - 2 * self.order
        for i in range(self.order, num_points + self.order):
            qL[i] = np.dot(weights[:, i], q_stencils[:, i])
        return qL, weights

    def rk_substep(self):

        # get the solution data
        g = self.precise_grid

        # comput flux at each point
        f = self.flux_function(g.u)

        # get maximum velocity
        alpha = np.max(abs(g.u))

        # Lax Friedrichs Flux Splitting
        fp = (f + alpha * g.u) / 2
        fm = (f - alpha * g.u) / 2

        fpr = g.scratch_array()
        fml = g.scratch_array()
        flux = g.scratch_array()

        # compute fluxes at the cell edges
        # compute f plus to the right
        fpr[1:], fp_weights = self.weno_new(fp[:-1])
        # compute f minus to the left
        # pass the data in reverse order
        fml[-1::-1], fm_weights = self.weno_new(fm[-1::-1])

        if self.record_actions is not None:
            action_weights = np.stack((fp_weights[:, self.ng-1:-(self.ng-1)], fm_weights[:, -(self.ng+1):self.ng-2:-1]))
            # resulting array is (fp, fm) X stencil X location
            # transpose to location X (fp, fm) X stencil
            action_weights = action_weights.transpose((2, 0, 1))

            if self.record_actions == "weno":
                self.action_history.append(action_weights)
            elif self.record_actions == "coef":
                # Same as in Standard WENO agent in agents.py.
                order = self.order
                a_mat = weno_coefficients.a_all[order]
                a_mat = np.flip(a_mat, axis=-1)
                combined_weights = a_mat[None, None, :, :] * action_weights[:, :, :, None]

                flux_weights = np.zeros((g.real_length() + 1, 2, self.order * 2 - 1))
                for sub_stencil_index in range(order):
                    flux_weights[:, :, sub_stencil_index:sub_stencil_index + order] += combined_weights[:, :, sub_stencil_index, :]
                self.action_history.append(flux_weights)

        # compute flux from fpr and fml
        flux[1:-1] = fpr[1:-1] + fml[1:-1]
        rhs = g.scratch_array()

        if self.eps > 0.0:
            R = self.eps * self.lap()
            rhs[1:-1] = 1 / g.dx * (flux[1:-1] - flux[2:]) + R[1:-1]
        else:
            rhs[1:-1] = 1 / g.dx * (flux[1:-1] - flux[2:])

        if self.source is not None:
            rhs[1:-1] += self.source.get_full()[1:-1]

        return rhs

    def lap(self):
        """
        Returns the Laplacian of g.u.

        This calculation relies on ghost cells, so make sure they have been filled before calling this.
        """

        gr = self.precise_grid
        u = gr.u

        lapu = gr.scratch_array()

        ib = gr.ilo - 1
        ie = gr.ihi + 1

        lapu[ib:ie + 1] = (u[ib - 1:ie] - 2.0 * u[ib:ie + 1] + u[ib + 1:ie + 2]) / gr.dx ** 2

        return lapu

    def update(self, dt, time):
        # Do Euler step, though rk_substep is separated so this can be converted to RK4.
        self.t = time
        euler_step = dt * self.rk_substep()

        self.precise_grid.u += euler_step
        self.precise_grid.update_boundary()

    def get_full(self):
        """ Downsample the precise solution to get coarse solution values. """

        grid = self.precise_grid.get_full()
        if self.extra_ghosts > 0:
            grid = grid[self.extra_ghosts:-self.extra_ghosts]

        # For each coarse cell, there are precise_scale precise cells, where the middle
        # cell corresponds to the same location as the coarse cell.
        middle = int(self.precise_scale / 2)
        return grid[middle::self.precise_scale]

    def get_real(self):
        return self.get_full()[self.ng:-self.ng]

    def reset(self, **init_params):
        self.precise_grid.reset(init_params)

MAX_ITERS = 1000
class ImplicitSolution(SolutionBase):

    def __init__(self, nx, ng, xmin, xmax, epsilon=1e-14):
        super().__init__(nx=nx, ng=ng, xmin=xmin, xmax=xmax)
        self.epsilon = epsilon
        self.iterate = None

    def update(self, dt, time):

        def iterate(old_u, time):
            new_u = self.amplitude * np.tanh(self.k * (self.x - 0.5 - old_u*time))
            return new_u

        def smooth_rare(u):
            return self.amplitude * np.tanh(self.k * (self.x - 0.5 - u*time))

        try:
            self.u = fixed_point(self.iterate, self.u, args=(time,))
        except Exception:
            print("Failed to converge.")
            #TODO handle this better

    def get_full(self):
        return self.u

    def get_real(self):
        return self.u[self.ng:-self.ng]

    def reset(self, **params):
        raise NotImplementedError()


#TODO account for xmin, xmax in case they're not 0 and 1.
class SmoothSineSolution(ImplicitSolution):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        def iterate(old_u, time):
            return self.amplitude * np.sin(2 * np.pi * (self.x - old_u * time))
        self.iterate = iterate

    def reset(self, A=1.0, **kwargs):
        self.amplitude = A

        self.u = self.amplitude*np.sin(2 * np.pi * self.x)

class SmoothRareSolution(ImplicitSolution):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        def iterate(old_u, time):
            return self.amplitude * np.tanh(self.k * (self.x - 0.5 - old_u*time))
        self.iterate = iterate

    def reset(self, A=1.0, k=1.0, **kwargs):
        self.amplitude = A
        self.k = k

        self.u = self.amplitude * np.tanh(self.k * (self.x - 0.5))
