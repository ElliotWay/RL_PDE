import numpy as np
import tensorflow as tf
from gym import spaces

from envs.burgers_env import AbstractBurgersEnv
from envs.plottable_env import Plottable2DEnv
from envs.weno_solution import lf_flux_split_nd, weno_sub_stencils_nd
from envs.weno_solution import tf_lf_flux_split, tf_weno_sub_stencils
from envs.weno_solution import WENOSolution
from util.softmax_box import SoftmaxBox
from util.misc import create_stencil_indexes


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
                shape=(num_x + 1, num_y, 2, 2*self.state_order-1),
                dtype=np.float64)
        y_rl_state = SoftmaxBox(low=-1e7, high=1e7,
                shape=(num_x, num_y + 1, 2, 2*self.state_order-1),
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
        num_x, num_y = self.grid.num_cells
        ghost_x, ghost_y = self.grid.num_ghosts

        # Lax Friedrichs flux splitting.
        (flux_left, flux_right), (flux_down, flux_up) = lf_flux_split_nd(flux, self.grid.space)

        # Trim vertical ghost cells from horizontally split flux. (Should this be part of flux
        # splitting?)
        flux_left = flux_left[:, ghost_y:-ghost_y]
        flux_right = flux_right[:, ghost_y:-ghost_y]
        right_stencil_indexes = create_stencil_indexes(stencil_size=self.state_order * 2 - 1,
                                                       num_stencils=num_x + 1,
                                                       offset=ghost_x - self.state_order)
        left_stencil_indexes = np.flip(right_stencil_indexes, axis=-1) + 1
        # Indexing is tricky here. It creates stencils on dimension 1, so we need to transpose them
        # to the last axis.
        #right_stencils = (flux_right.transpose()[:, right_stencil_indexes]).transpose([1,0,2])
        #left_stencils = (flux_left.transpose()[:, left_stencil_indexes]).transpose([1,0,2])
        right_stencils = flux_right[:, right_stencil_indexes].transpose([0,2,1])
        left_stencils = flux_left[:, left_stencil_indexes].transpose([0,2,1])
        horizontal_state = np.stack([left_stencils, right_stencils], axis=2)

        flux_down = flux_down[ghost_x:-ghost_x, :]
        flux_up = flux_up[ghost_x:-ghost_x, :]
        up_stencil_indexes = create_stencil_indexes(stencil_size=self.state_order * 2 - 1,
                                                    num_stencils=num_y + 1,
                                                    offset=ghost_y - self.state_order)
        down_stencil_indexes = np.flip(up_stencil_indexes, axis=-1) + 1
        up_stencils = flux_up[:, up_stencil_indexes]
        down_stencils = flux_down[:, down_stencil_indexes]
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
        left_sub_stencils = weno_sub_stencils_nd(left_stencils, self.weno_order)
        right_sub_stencils = weno_sub_stencils_nd(right_stencils, self.weno_order)
        down_sub_stencils = weno_sub_stencils_nd(down_stencils, self.weno_order)
        up_sub_stencils = weno_sub_stencils_nd(up_stencils, self.weno_order)

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

        cell_size_x, cell_size_y = self.grid.cell_size

        step = (  (horizontal_flux_reconstructed[:-1, :]
                    - horizontal_flux_reconstructed[1:, :]) / cell_size_x
                + (vertical_flux_reconstructed[:, :-1]
                    - vertical_flux_reconstructed[:, 1:]) / cell_size_y
                )

        return step

    @tf.function
    def tf_prep_state(self, state):
        # Equivalent to _prep_state() above.

        # state (the real physical state) does not have ghost cells, but agents operate on a stencil
        # that can spill beyond the boundary, so we need to add ghost cells to create the rl_state.

        # TODO Same as 1D version, this should handle boundary as a parameter instead of using the
        # current grid value.
        full_state = self.grid.tf_update_boundary(state, self.grid.boundary)
        flux = 0.5 * (full_state ** 2)
        num_x, num_y = self.grid.num_cells
        ghost_x, ghost_y = self.grid.num_ghosts

        (flux_left, flux_right), (flux_down, flux_up) = tf_lf_flux_split(flux, full_state)

        flux_left = flux_left[:, ghost_y:-ghost_y]
        flux_right = flux_right[:, ghost_y:-ghost_y]
        right_stencil_indexes = create_stencil_indexes(stencil_size=self.state_order * 2 - 1,
                                                       num_stencils=num_x + 1,
                                                       offset=ghost_x - self.state_order)
        left_stencil_indexes = np.flip(right_stencil_indexes, axis=-1) + 1
        right_stencils = tf.transpose(tf.gather(flux_right, right_stencil_indexes), [0,2,1])
        left_stencils = tf.transpose(tf.gather(flux_left, left_stencil_indexes), [0,2,1])
        horizontal_state = tf.stack([left_stencils, right_stencils], axis=2)

        flux_down = flux_down[ghost_x:-ghost_x, :]
        flux_up = flux_up[ghost_x:-ghost_x, :]
        up_stencil_indexes = create_stencil_indexes(stencil_size=self.state_order * 2 - 1,
                                                    num_stencils=num_y + 1,
                                                    offset=ghost_y - self.state_order)
        down_stencil_indexes = np.flip(up_stencil_indexes, axis=-1) + 1
        up_stencils = tf.gather(flux_up, up_stencil_indexes)
        down_stencils = tf.gather(flux_down, down_stencil_indexes)
        vertical_state = np.stack([down_stencils, up_stencils], axis=2)

        rl_state = (horizontal_state, vertical_state)

        return rl_state

    @tf.function
    def tf_integrate(self, args):
        # (Mostly) equivalent to _rk_substep() above.
        real_state, rl_state, rl_action = args

        # Note that real_state does not contain ghost cells here, but rl_state DOES (and rl_action has
        # weights corresponding to the ghost cells).

        horizontal_stencils, vertical_stencils = rl_state
        left_stencils = horizontal_stencils[:, :, 0]
        right_stencils = horiztonal_stencils[:, :, 1]
        down_stencils = vertical_stencils[:, :, 0]
        up_stencils = vertical_stencils[:, :, 1]

        horizontal_weights, vertical_weights = rl_action
        left_weights = horizontal_weights[:, :, 0]
        right_weights = horiztonal_weights[:, :, 1]
        down_weights = vertical_weights[:, :, 0]
        up_weights = vertical_weights[:, :, 1]

        left_reconstructed = tf.reduce_sum(left_weights 
                                * tf_weno_sub_stencils(left_stencils, self.weno_order), axis=-1)
        right_reconstructed = tf.reduce_sum(right_weights
                                * tf.weno_sub_stencils(right_stencils, self.weno_order), axis=-1)
        down_reconstructed = tf.reduce_sum(down_weights
                                * tf.weno_sub_stencils(down_stencils, self.weno_order), axis=-1)
        up_reconstructed = tf.reduce_sum(up_weights
                                * tf.weno_sub_stencils(up_stencils, self.weno_order), axis=-1)

        horiztonal_flux_reconstructed = left_reconstructed + right_reconstructed
        vertical_flux_reconstructed = down_reconstructed + up_reconstructed

        cell_size_x, cell_size_y = self.grid.cell_size

        step = (  (horizontal_flux_reconstructed[:-1, :]
                    - horizontal_flux_reconstructed[1:, :]) / cell_size_x
                + (vertical_flux_reconstructed[:, :-1]
                    - vertical_flux_reconstructed[:, 1:]) / cell_size_y
                )

        step = self.dt * step

        if self.nu != 0.0:
            step += self.dt * self.nu * self.grid.tf_laplacian(real_state)
        if self.source != None:
            raise NotImplementedError("External source has not been implemented"
                    + " in global backprop.")

        new_state = real_state + step
        return new_state


