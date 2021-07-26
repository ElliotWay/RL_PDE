import numpy as np
import tensorflow as tf
from gym import spaces

from envs.burgers_env import AbstractBurgersEnv
from envs.plottable_env import Plottable2DEnv
from util.softmax_box import SoftmaxBox

from util.misc import AxisSlice

#TODO Put these functions somewhere else.
def weno_sub_stencils_nd(order, stencils_array):
    """
    Interpolate sub-stencils in an ndarray of stencils.

    An ndarray of stencils:
    [spatial dimensions... X stencil size]
    ->
    An ndarray of interpolated sub-stencils:
    [spatial dimensions... X num sub-stencils]
    """
    # These weights have shape order X order (i.e. num stencils * stencil size).
    a_mat = weno_coefficients.a_all[order]

    # These weights are "backwards" in the original formulation.
    # This is easier in the original formulation because we can add the k for our kth stencil to the index,
    # then subtract by a variable amount to get each value, but there's no need to do that here, and flipping
    # it back around makes the expression simpler.
    a_mat = np.flip(a_mat, axis=-1)

    sub_stencil_indexes = create_stencil_indexes(stencil_size=order, num_stencils=order)
    sub_stencils = AxisSlice(stencils_array, -1)[sub_stencil_indexes]

    interpolated = np.sum(a_mat * sub_stencils, axis=-1)
    return interpolated

class WENOBurgers2DEnv(AbstractBurgersEnv, Plottable2DEnv):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        num_x, num_y = self.grid.num_cells

        # Define spaces.
        x_actions = SoftmaxBox(low=0.0, high=1.0,
                # num x interfaces X num y cells X (+,-) X num substencils
                shape=(num_x + 1, num_y, 2, self.weno_order),
                dtype=np.float64)
        y_actions = SoftmaxBox(low=0.0, high=1.0,
                shape=(num_x, num_y + 1, 2, self.weno_order),
                dtype=np.float64)
        self.action_space = spaces.Tuple((x_actions, y_actions))
        
        x_rl_state = spaces.Box(low=-1e7, high=1e7,
                # num x interfaces X num y cells X (+,-) X stencil size
                shape=(num_x + 1, num_y, 2, 2*self.state_order+1),
                dtype=np.float64)
        y_rl_state = SoftmaxBox(low-1e7, high=1e7,
                shape=(num_x, num_y + 1, 2, 2*self.state_order+1),
                dtype=np.float64)
        self.observation_space = spaces.Tuple((x_rl_state, y_rl_state))

        # Set solution(s) to record history and actions.
        self.solution.set_record_state(True)
        if self.weno_solution is not None:
            self.weno_solution.set_record_state(True)

        if self.weno_solution is not None:
            self.weno_solution.set_record_actions("weno")
        elif not isinstance(self.solution, WENOSolution):
            self.solution.set_record_actions("weno")

        # Something like this for action labels?
        #self._action_labels = ["$w^{}_{}$".format(sign, num) for sign in ['+', '-']
                                    #for num in range(1, self.weno_order+1)]

    def _prep_state(self):
        u_values = self.grid.get_full()
        flux = self.burgers_flux(u_values)

        # Lax Friedrichs Flux Splitting
        x_alpha = np.max(np.abs(u_values), axis=0)
        y_alpha = np.max(np.abs(u_values), axis=1)
        flux_left = (flux - x_alpha * u_values) / 2
        flux_right = (flux + x_alpha * u_values) / 2
        flux_down = (flux - y_alpha * u_values) / 2
        flux_up = (flux + y_alpha * u_values) / 2

        num_x, num_y = self.grid.num_cells
        ghost_x, ghost_y = self.grid.num_ghosts

        right_stencil_indexes = create_stencil_indexes(stencil_size=self.state_order * 2 - 1,
                                                       num_stencils=num_x + 1,
                                                       offset=ghost_x - self.state_order)
        left_stencil_indexes = np.flip(right_stencil_indexes, axis=-1) + 1
        # Indexing is tricky here. I couldn't find a way without transposing and then transposing
        # back. (It might be possible, though.)
        right_stencils = (flux_right.transpose()[:, indexes]).transpose([1,0,2])
        left_stencils = (flux_left.transpose()[:, indexes]).transpose([1,0,2])
        horizontal_state = np.stack([left_stencils, right_stencils], axis=2)

        up_stencil_indexes = create_stencil_indexes(stencil_size=self.state_order * 2 - 1,
                                                    num_stencils=num_y + 1,
                                                    offset=ghost_y - self.state_order)
        down_stencil_indexes = np.flip(up_stencil_indexes, axis=-1) + 1
        up_stencils = flux_up[:, indexes]
        down_stencil = flux_down[:, indexes]
        vertical_state = np.stack([down_stencils, up_stencils], axis=2)

        state = (horizontal_state, vertical_state)

        self.current_state = state
        return state

    def _rk_substep(self, action):

        x_state, y_state = self.current_state

        left_stencils = x_state[:, :, 0, :]
        right_stencils = x_state[:, :, 1, :]
        down_stencils = y_state[:, :, 0, :]
        up_stencils = y_state[:, :, 1, :]

        #TODO I think we need to offset the sub_stencil indexes by the state_order - weno_order.
        left_sub_stencils = weno_sub_stencils_nd(self.weno_order, left_stencils)
        right_sub_stencils = weno_sub_stencils_nd(self.weno_order, right_stencils)
        down_sub_stencils = weno_sub_stencils_nd(self.weno_order, down_stencils)
        up_sub_stencils = weno_sub_stencils_nd(self.weno_order, up_stencils)

        x_action, y_action = action
        left_action = x_action[:, :, 0, :]
        right_action = x_action[:, :, 1, :]
        down_action = y_action[:, :, 0, :]
        up_action = y_action[:, :, 1, :]

        left_flux_reconstructed = np.sum(left_action * left_sub_stencils, axis=-1)
        right_flux_reconstructed = np.sum(right_action * right_sub_stencils, axis=-1)
        down_flux_reconstructed = np.sum(down_action * down_sub_stencils, axis=-1)
        up_flux_reconstructed = np.sum(up_action * up_sub_stencils, axis=-1)

        horizontal_flux_reconstructed = left_flux_reconstructed + right_flux_reconstructed
        vertical_flux_reconstructed = down_flux_reconstructed + up_flux_reconstructed

        cell_size_x, cell_size_y = g.cell_size

        step = (  (horizontal_flux_reconstructed[:-1, :]
                    - horizontal_flux_reconstructed[1:, :]) / cell_size_x
                + (vertical_flux_reconstructed[:, :-1]
                    - vertical_flux_reconstructed[:, 1:]) / cell_size_y
                )

        return step
