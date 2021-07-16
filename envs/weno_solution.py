import numpy as np

from envs.solutions import SolutionBase
import envs.weno_coefficients as weno_coefficients
from envs.grid import AbstractGrid, create_grid
from envs.grid1d import Grid1d
from util.misc import create_stencil_indexes


# Should be possible to convert this to ND instead of 2D.
# rk_substep needs to be generalized.
class PreciseWENOSolution2D(SolutionBase):
    def __init__(self, base_grid, init_params,
            precise_order, precise_scale,
            flux_function,
            eps=0.0, source=None,
            record_state=False, record_actions=None):
        super().__init__(base_grid.num_cells, base_grid.num_ghosts,
                base_grid.min_value, base_grid.max_value, record_state=record_state)
 
        assert (precise_scale % 2 == 1), "Precise scale must be odd for easier downsampling."

        self.precise_scale = precise_scale

        self.precise_num_cells = [precise_scale * nx for nx in base_grid.num_cells]
        self.precise_num_ghosts = []
        self.extra_ghosts = []
        for ng in base_grid.num_ghosts:
            if precise_order + 1 > precise_scale * ng:
                precise_ng = precise_order + 1
                self.precise_num_ghosts.append(precise_ng)
                self.extra_ghosts.append(precise_ng - (precise_scale*ng))
            else:
                self.precise_num_ghosts.append(precise_scale * ng)
                self.extra_ghosts.append(0)

        if 'boundary' in init_params:
            self.boundary = init_params['boundary']
        else:
            self.boundary = None
        if 'init_type' in init_params:
            self.init_type = init_params['init_type']
        else:
            self.init_type = None

        self.precise_grid = create_grid(len(self.precise_num_cells),
                self.precise_num_cells, self.precise_num_ghosts,
                base_grid.min_value, base_grid.max_value,
                init_type=self.init_type, boundary=self.boundary)

        self.flux_function = flux_function
        self.eps = eps
        self.precise_order = precise_order
        self.source = source

        self.record_actions = record_actions
        self.action_history = []

        self.use_rk4 = False
 
    def set_rk4(self, use_rk4):
        self.use_rk4 = use_rk4

    def is_recording_actions(self):
        return (self.record_actions is not None)
    def set_record_actions(self, record_mode):
        self.record_actions = record_mode
    def get_action_history(self):
        return self.action_history

    def weno_sub_stencils(self, stencils_batch):
        order = self.precise_order
        a_mat = weno_coefficients.a_all[order]

        # These weights are "backwards" in the original formulation.
        # This is easier in the original formulation because we can add the k for our kth stencil to the index,
        # then subtract by a variable amount to get each value, but there's no need to do that here, and flipping
        # it back around makes the expression simpler.
        a_mat = np.flip(a_mat, axis=-1)

        sub_stencil_indexes = create_stencil_indexes(stencil_size=order, num_stencils=order)

        sub_stencils = np.sum(a_mat * stencils_batch[:, sub_stencil_indexes], axis=-1)
        return sub_stencils

    def weno_weights(self, stencils_batch):
        # Adapted from agents.py#StandardWENOAgent.
        order = self.precise_order

        C = weno_coefficients.C_all[order]
        sigma = weno_coefficients.sigma_all[order]
        epsilon = 1e-16

        sliding_window_indexes = np.arange(order)[None, :] + np.arange(order)[:, None]

        # beta refers to the smoothness indicator.
        # To my best understanding, beta is calculated based on an approximation of the square of the values
        # in q, so we multiply every pair of values in a sub-stencil together, weighted appropriately for
        # the approximation by sigma.

        sub_stencils = stencils_batch[:, sliding_window_indexes]

        # Flipped because in original formulation we subtract from last index instead of adding from first,
        # and sigma and C weights assume this ordering.
        sub_stencils_flipped = np.flip(sub_stencils, axis=-1)

        # Insert extra empty dimensions so numpy broadcasting produces
        # the desired outer product along only the intended dimensions.
        q_squared = sub_stencils_flipped[:, :, None, :] * sub_stencils_flipped[:, :, :, None]

        # Note that sigma is made up of triangular matrices, so the second copy of each pair is weighted by 0.
        beta = np.sum(sigma * q_squared, axis=(-2, -1))

        alpha = C / (epsilon + beta ** 2)

        # We need the [:, None] so numpy broadcasts to the sub-stencil dimension, instead of the stencil dimension.
        weights = alpha / (np.sum(alpha, axis=-1)[:, None])

        return weights

    def weno_reconstruct(self, stencils):
        """
        Compute WENO reconstruction.

        Parameters
        ----------
        q : ndarray (number of interfaces X stencil size)
          Stencils of scalar data to reconstruct. Probably fluxes.

        Returns
        -------
        Reconstructed data at the interfaces.

        """
        weights = self.weno_weights(stencils)
        sub_stencils = self.weno_sub_stencils(stencils)

        reconstructed = np.sum(weights * sub_stencils, axis=-1)
        return reconstructed, weights

    def rk_substep(self):
        g = self.precise_grid
        g.update_boundary()
        order = self.precise_order
        num_x, num_y = g.num_cells
        num_ghost_x, num_ghost_y = g.num_ghosts
        cell_size_x, cell_size_y = g.cell_size

        # compute flux at each point
        flux = self.flux_function(g.space)

        # Recompute flux over every row and every column.
        vertical_flux_reconstructed = np.zeros((num_x, num_y+1)) # index a,b is actually a,b-1/2
        horizontal_flux_reconstructed = np.zeros((num_x+1, num_y)) # index a,b is actually a-1/2, b
        if self.record_actions is not None:
            vertical_actions = np.zeros((num_x, num_y+1, 2, order))
            horizontal_actions = np.zeros((num_x+1, num_y, 2, order))
        #TODO: Vectorize further? weno_reconstruct() is already vectorized,
        # but could reasonably be extended to apply to the entire space at once.
        up_stencil_indexes = create_stencil_indexes(stencil_size=order * 2 - 1,
                                                    num_stencils=num_y + 1,
                                                    offset=num_ghost_y - order)
        down_stencil_indexes = np.flip(up_stencil_indexes, axis=-1) + 1
        for x_index in range(num_x):
            grid_row = g.space[x_index, :]
            flux_row = flux[x_index, :]
            alpha = np.max(grid_row)

            flux_up = (flux_row + alpha * grid_row) / 2
            flux_down = (flux_row - alpha * grid_row) / 2

            up_stencils = flux_up[up_stencil_indexes]
            up_reconstructed, up_weights = self.weno_reconstruct(up_stencils)

            down_stencils = flux_down[down_stencil_indexes]
            down_reconstructed, down_weights = self.weno_reconstruct(down_stencils)

            if self.record_actions is not None:
                vertical_actions[x_index, :, 0, :] = up_weights
                vertical_actions[x_index, :, 1, :] = down_weights
            
            vertical_flux_reconstructed[x_index, :] = up_reconstructed + down_reconstructed

        right_stencil_indexes = create_stencil_indexes(stencil_size=order * 2 - 1,
                                                       num_stencils=num_x + 1,
                                                       offset=num_ghost_x - order)
        left_stencil_indexes = np.flip(right_stencil_indexes, axis=-1) + 1
        for y_index in range(num_y):
            grid_col = g.space[:, y_index]
            flux_col = flux[:, y_index]
            # get maximum velocity
            alpha = np.max(grid_col)

            # Lax Friedrichs Flux Splitting
            flux_right = (flux_col + alpha * grid_col) / 2
            flux_left = (flux_col - alpha * grid_col) / 2

            right_stencils = flux_right[right_stencil_indexes]
            right_reconstructed, right_weights = self.weno_reconstruct(right_stencils)

            left_stencils = flux_left[left_stencil_indexes]
            left_reconstructed, left_weights = self.weno_reconstruct(left_stencils)
            
            if self.record_actions is not None:
                horizontal_actions[:, y_index, 0, :] = right_weights
                horizontal_actions[:, y_index, 1, :] = left_weights

            horizontal_flux_reconstructed[:, y_index] = right_reconstructed + left_reconstructed

        if self.record_actions is not None:
            if self.record_actions == "weno":
                self.action_history.append((vertical_actions, horizontal_actions))
            elif self.record_actions == "coef":
                raise NotImplementedError()
                # This corresponds to e.g. SplitFluxBurgersEnv.
                # Still need to convert this to 2D.
                # Related to Standard WENO agent in agents.py.
                #a_mat = weno_coefficients.a_all[order]
                #a_mat = np.flip(a_mat, axis=-1)
                #combined_weights = a_mat[None, None, :, :] * action_weights[:, :, :, None]

                #flux_weights = np.zeros((g.real_length() + 1, 2, self.order * 2 - 1))
                #for sub_stencil_index in range(order):
                    #flux_weights[:, :, sub_stencil_index:sub_stencil_index + order] += combined_weights[:, :, sub_stencil_index, :]
                #self.action_history.append(flux_weights)
            else:
                raise Exception("Unrecognized action type: '{}'".format(self.record_actions))

        rhs = (  (vertical_flux_reconstructed[:, :-1]
                    - vertical_flux_reconstructed[:, 1:]) / cell_size_y
               + (horizontal_flux_reconstructed[:-1, :]
                    - horizontal_flux_reconstructed[1:, :]) / cell_size_x
               )

        return rhs

    def _update(self, dt, time):
        u_start = np.array(self.precise_grid.get_real())
        if self.use_rk4:
            k1 = dt * self.rk_substep()
            self.precise_grid.set(u_start + (k1 / 2))

            k2 = dt * self.rk_substep()
            self.precise_grid.set(u_start + (k2 / 2))

            k3 = dt * self.rk_substep()
            self.precise_grid.set(u_start + k3)

            k4 = dt * self.rk_substep()
            full_step = (k1 + 2*(k2 + k3) + k4) / 6

        else: #Euler
            full_step = dt * self.rk_substep()

        if self.eps > 0.0:
            if self.use_rk4:
                self.precise_grid.set(u_start)
            R = self.eps * self.laplacian()
            full_step += dt * R[num_ghost_x:-num_ghost_x, num_ghost_y:-num_ghost_y]

        if self.source is not None:
            step += dt * self.source.get_real()

        self.precise_grid.set(u_start + full_step)
        self.precise_grid.update_boundary()

    def laplacian(self):
        """
        Returns the Laplacian of g.u.

        This calculation relies on ghost cells, so make sure they have been filled before calling this.
        """
        #TODO: rewrite this for 2D.

        gr = self.precise_grid
        u = gr.u

        lapu = gr.scratch_array()

        ib = gr.ilo - 1
        ie = gr.ihi + 1

        lapu[ib:ie + 1] = (u[ib - 1:ie] - 2.0 * u[ib:ie + 1] + u[ib + 1:ie + 2]) / gr.dx ** 2

        return lapu

    def get_full(self):
        """ Downsample the precise solution to get coarse solution values. """

        grid = self.precise_grid.get_full()

        if any([xg > 0 for xg in self.extra_ghosts]):
            extra_ghost_slice = tuple([slice(xg, -xg) for xg in self.extra_ghosts])
            grid = grid[extra_ghost_slice]

        # For each coarse cell, there are precise_scale precise cells, where the middle
        # cell corresponds to the same location as the coarse cell.
        middle = int(self.precise_scale / 2)
        middle_cells_slice = tuple([slice(middle, None, self.precise_scale) for _ in
            self.num_cells])
        return grid[middle_cells_slice]

    def get_real(self):
        real_slice = tuple([slice(ng, -ng) for ng in self.num_ghosts])
        return self.get_full()[real_slice]

    def _reset(self, init_params):
        self.precise_grid.reset(init_params)
        self.action_history = []

    def set(self, real_values):
        """ Force set the current grid. Will make the state/action history confusing. """
        self.precise_grid.set(real_values)

     

class PreciseWENOSolution(SolutionBase):

    def __init__(self, nx, ng, xmin, xmax,
                 precise_order, precise_scale, init_type, boundary, flux_function,
                 eps=0.0, source=None,
                 record_state=False, record_actions=None):
        super().__init__(nx, ng, xmin, xmax)

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

        self.record_actions = record_actions
        self.action_history = []

        self.use_rk4 = False

    def set_rk4(self, use_rk4):
        self.use_rk4 = use_rk4

    def is_recording_actions(self):
        return (self.record_actions is not None)
    def set_record_actions(self, record_mode):
        self.record_actions = record_mode
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
        g.update_boundary()


        # compute flux at each point
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
            else:
                raise Exception("Unrecognized action type: '{}'".format(self.record_actions))

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

    def _update(self, dt, time):
        if self.use_rk4:
            u_start = np.array(self.precise_grid.get_full())
            
            k1 = dt * self.rk_substep()
            self.precise_grid.u = u_start + (k1 / 2)

            k2 = dt * self.rk_substep()
            self.precise_grid.u = u_start + (k2 / 2)

            k3 = dt * self.rk_substep()
            self.precise_grid.u = u_start + k3

            k4 = dt * self.rk_substep()
            k4_step = (k1 + 2*(k2 + k3) + k4) / 6
            self.precise_grid.u = u_start + k4_step

        else:
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

    def _reset(self, init_params):
        self.precise_grid.reset(init_params)
        self.action_history = []

    def set(self, real_values):
        """ Force set the current grid. Will make the state/action history confusing. """
        self.precise_grid.set(real_values)


if __name__ == "__main__":
    import os
    import shutil
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    from matplotlib import cm
    from matplotlib.ticker import LinearLocator
    from envs.grid2d import Grid2d

    order = 2
    ng = order + 1

    base_grid = Grid2d((128, 128), ng, 0.0, 1.0)
    flux_function = lambda x: 0.5 * x ** 2
    sol = PreciseWENOSolution2D(base_grid, {}, order, 1, flux_function)
    sol.reset({'init_type': '1d-smooth_sine-y'})

    timestep = 0.0004
    time = 0.0

    save_dir = "weno2d_test"
    try:
        os.makedirs(save_dir)
    except FileExistsError:
        # Directory already exists, overwrite.
        shutil.rmtree(save_dir)
        os.makedirs(save_dir)

    for step in range(250):
        if step % 10 == 0:
            print("{}...".format(step), end='', flush=True)
            fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
            x, y = np.meshgrid(sol.real_x, sol.real_y)
            z = sol.get_real()
            surface = ax.plot_surface(x, y, z, cmap=cm.viridis,
                    linewidth=0, antialiased=False)
            #ax.set_zlim(-0.25, 1.25)
            ax.zaxis.set_major_locator(LinearLocator(10))
            #ax.zaxis.set_major_formatter(StrMethodFormatter('{x:0.2f}'))
            fig.colorbar(surface, shrink=0.5, aspect=5)

            ax.set_title("t={:.4f}".format(time))

            filename = os.path.join(save_dir, "plot_{:03d}.png".format(step))
            plt.savefig(filename)
            plt.close(fig)
        time += timestep
        sol.update(timestep, time)
    print("Done")
