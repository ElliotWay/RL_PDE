import os
from collections import OrderedDict
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Layer

from models import GlobalModel
from models.builder import get_optimizer
from models.net import PolicyNet, FunctionWrapper
import envs.weno_coefficients as weno_coefficients
from util.misc import create_stencil_indexes
from util.serialize import save_to_zip, load_from_zip
from util.function_dict import tensorflow_fn
from util.serialize import save_to_zip, load_from_zip

SUPPORTED_BOUNDARIES = ["outflow", "periodic"]

class GlobalBackpropModel(GlobalModel):
    """
    Model that uses the "global backprop" idea, that is, we construct a recurrent network that
    unrolls to the entire length of an episode, with the transition and reward functions built into
    the network. This allows us to backpropagate over the entire length of space and time, as the
    transition and reward functions are known and differentiable. We can also optimize the rewards
    directly, instead of using e.g. the policy gradient.

    Unlike our other models, it doesn't make sense to train with a set of trajectories, as the
    global model has trajectories as part of its internal workings. Instead, the only training
    input is the initial state, so the train function is changed to train(initial_state) instead.
    The predict method, on the other hand, is the same, as the global must still have the local
    model built in.
    """
    def __init__(self, env, args, dtype=tf.float64):
        """
        Create the GlobalBackprop Model.
        Note that as of writing this, the env needs to be specifically an AbstractPDEEnv - a general
        gym.Env instance won't work.
        """
        self.dtype = dtype
        self.env = env

        # Stop Tensorflow from eating up the entire GPU (if we're using one).
        tf_config = tf.ConfigProto()
        tf_config.gpu_options.allow_growth = True

        self.session = tf.Session(config=tf_config)

        self._policy_ready = False
        self._training_ready = False
        self._loading_ready = False

        # Construction for this model is now lazy. The model is not set up to train until train()
        # is called, the policy is not built until predict() or train() is called, and the model is
        # not ready to load until load() or train() is called. Lazy construction like this is
        # useful for testing in particular - this means that we can construct only the part of the
        # model needed for testing without building in all of the details needed to build the
        # entire model, e.g. nx and episode_length.

        self.preloaded = False

        self.args = args # Not good to save global copy of args, but we need it to do lazy model
                         # construction.

        if 'log_freq' in args:
        # if args.log_freq doesn't exist, we're probably in testing so doesn't matter.
            self.log_freq = args.log_freq
        else:
            self.log_freq = None

    def setup_training(self):
        if not self._policy_ready:
            self.setup_policy()
        if not self._loading_ready:
            self.setup_loading()

        if not self._training_ready:
            obs_adjust = tensorflow_fn(self.args.obs_scale)
            action_adjust = tensorflow_fn(self.args.action_scale)
            self.wrapped_policy = FunctionWrapper(layer=self.policy, input_fn=obs_adjust,
                    output_fn=action_adjust)
            # TODO: make function wrapping more sane
            # You may be wondering why we apply the normalizing functions here for the RNN,
            # but not when self.policy is declared, meaning that calling predict does not
            # normalize the state and returned action. The discrepancy is in that the EMI
            # handles input and output to the policy during testing, so the normalizing functions
            # are applied there instead of in setup_policy.
            # This does not make a lot of sense and should be changed. Ideally, this behavior
            # is entirely controlled by the EMI - but how do we move it there?

            # This creates separate networks for each boundary condition.
            # This is necessary because building a condition into the network, while theoretically
            # possible, incurs an intractable cost.
            # Instead we separate each batch into a 'sub-batch' for each boundary condition.
            self.init_state_ph_dict = {}
            self.state_dict = {}
            self.action_dict = {}
            self.reward_dict = {}
            self.loss_dict = {}
            self.gradients_dict = {}

            # Create all possible combinations of boundary conditions in different dimensions.
            # So with [outflow, periodic] there are 4 combinations in 2 dimensions and 8 in 3.
            self.boundary_combinations = [tuple(combination) for combination in
                        np.stack([dimension.ravel() for dimension in
                            np.meshgrid(*([SUPPORTED_BOUNDARIES] * self.env.dimensions),
                                            indexing='xy')],
                            axis=-1)]
            # Future note: if there are simply too many to enumerate when we're only using
            # a few combinations, we could declare these lazily instead. This will cost the
            # debugging advantages of finalizing the graph.
            for boundary in self.boundary_combinations:
                params = {'boundary': boundary}
                self.env.reset(params)
                boundary_name = '_'.join(boundary)

                cell = IntegrateCell(
                        prep_state_fn=self.env.tf_prep_state,
                        policy_net=self.wrapped_policy,
                        integrate_fn=self.env.tf_integrate,
                        reward_fn=self.env.tf_calculate_reward,
                        )
                rnn = IntegrateRNN(cell, name="rnn_".format(boundary_name))

                # initial_state_ph is the REAL physical initial state
                # (It should not contain ghost cells - those should be handled by the prep_state function
                # that converts real state to rl state.)
                initial_state_ph = tf.placeholder(dtype=self.dtype,
                        shape=(None, self.env.grid.vec_len) + self.env.grid.num_cells,
                        name="init_real_state_{}".format(boundary_name))
                self.init_state_ph_dict[boundary] = initial_state_ph

                # Could restrict steps to something smaller.
                # This loses some of the benefits of RL long-term planning, but may be necessary in
                # high dimensional environments.
                state, action, reward = rnn(initial_state_ph, num_steps=self.args.ep_length)
                self.state_dict[boundary] = state
                self.action_dict[boundary] = action
                self.reward_dict[boundary] = reward

                # Check reward shape - should have a reward for each timestep, batch,
                # (optional vector part,) and physical location.
                #TODO Temporary hack - replace when vector EMI is implemented. -Elliot
                #assert len(reward[0].shape) == (3 if self.env.grid.vec_len > 1 else 2) + self.env.dimensions
                assert len(reward[0].shape) == (2 if self.env.grid.vec_len > 1 else 2) + self.env.dimensions
                # Sum over the timesteps in the trajectory (axis 0),
                # then average over the batch and each location and the batch (axes 1 and the rest).
                # Also average over each reward part (e.g. each dimension).
                loss = -tf.reduce_mean([tf.reduce_mean(tf.reduce_sum(reward_part, axis=0)) for
                                                reward_part in reward])
                self.loss_dict[boundary] = loss

                gradients = tf.gradients(loss, self.policy_params)
                self.gradients_dict[boundary] = gradients

            # The gradients for each boundary are passed out of tensorflow, where they are
            # combined and weighted, then passed back into tensorflow.
            # (This is necessary because we may not have samples in each boundary condition, and
            # Tensorflow doesn't work well with empty Tensors.)
            gradients_0 = self.gradients_dict[self.boundary_combinations[0]]
            self.gradients_ph_list = [tf.placeholder(dtype=grad.dtype, shape=grad.shape,
                        name="weighted_gradients_{}".format(i)) for i, grad in
                        enumerate(gradients_0)]
            self.grads = list(zip(self.gradients_ph_list, self.policy_params))

            # Declare this once - otherwise we add to the graph every time,
            # and it won't be garbage collected.
            # Also needs to be before we finalize the graph.
            self.params_nan_check = {param.name: tf.reduce_any(tf.is_nan(param))
                                        for param in self.policy_params}

            self.optimizer = get_optimizer(self.args)
            self.train_policy = self.optimizer.apply_gradients(self.grads)

            if not self.preloaded:
                tf.global_variables_initializer().run(session=self.session)

            self.session.graph.finalize()

            self._training_ready = True

    def setup_policy(self):
        if not self._policy_ready:
            #TODO This is the kind of thing that ought to be handled by the EMI.
            # Can we extract this responsibility from the global model?
            # Notably, the 2D vs 1D responibility IS being handled by the EMI.
            # Surely there is a way to handle the spatial dimension there as well? At least for
            # this part?
            action_shape = self.env.action_space.shape[1:]

            #TODO Use ReLU? A field in args with ReLU as the default might make sense.
            # Note: passing a name to the constructor of a Keras Layer has the effect
            # of putting everything in that layer in the scope of that name.
            # tf.variable_scope does not play well with Keras.
            self.policy = PolicyNet(layers=self.args.layers, action_shape=action_shape,
                    activation_fn=tf.nn.relu, name="policy", dtype=self.dtype)

            # Direct policy input and output used in predict() method during testing.
            self.policy_input_ph = tf.placeholder(dtype=self.dtype,
                    shape=(None,) + self.env.observation_space.shape[1:], name="policy_input")
            self.policy_output = self.policy(self.policy_input_ph)
            # See note in setup_training for why we do not apply normalizing functions here.

            self.policy_params = self.policy.weights

            self._policy_ready = True

    def setup_loading(self):
        if not self._policy_ready:
            self.setup_policy()

        if not self._loading_ready:
            self.load_op = {}
            self.load_ph = {}
            for param in self.policy_params:
                clean_name = ''.join(x if x not in "\0\ \t\n\\/:=.*\"\'<>|?" else '_'
                                        for x in str(param.name))
                placeholder = tf.placeholder(dtype=param.dtype, shape=param.shape, 
                        name="load_{}".format(clean_name))
                self.load_op[param] = param.assign(placeholder)
                self.load_ph[param] = placeholder

            self._loading_ready = True


    def train(self, initial_state, init_params):
        if not self._training_ready:
            self.setup_training()

        for params in init_params:
            if type(params['boundary']) is str:
                params['boundary'] = tuple((params['boundary'],) * self.env.dimensions)
            if not params['boundary'] in self.boundary_combinations:
                raise NotImplementedError("GlobalBackpropModel:"
                        + " Cannot train on {} boundary".format(params['boundary'])
                        + " Only the following boundary conditions are supported:"
                        + "\n{}".format(SUPPORTED_BOUNDARIES))

        # Separate initial_state into sub-batches for each boundary condition.
        index_dict = {}
        feed_dict = {}
        included_boundaries = []
        for boundary in self.boundary_combinations:
            indexes = [i for i, params in enumerate(init_params) if params['boundary'] == boundary]
            if len(indexes) > 0:
                included_boundaries.append(boundary)
                index_dict[boundary] = indexes
                feed_dict[self.init_state_ph_dict[boundary]] = initial_state[indexes]
                #print("{} has initial state with shape {}".format(boundary,
                    #feed_dict[self.init_state_ph_dict[boundary]].shape))
                #print("indexes are {}".format(indexes))

        # Get the gradients (but only for the boundaries actually being used).
        def subdict(d, keys):
            return {k: v for k, v in d.items() if k in keys}
        gradients_dict, loss_dict, reward_dict, action_dict, state_dict = self.session.run(
                [subdict(self.gradients_dict, included_boundaries),
                    subdict(self.loss_dict, included_boundaries),
                    subdict(self.reward_dict, included_boundaries),
                    subdict(self.action_dict, included_boundaries),
                    subdict(self.state_dict, included_boundaries)
                    ],
                feed_dict=feed_dict)

        # Compute gradients weighted by the proportion of each boundary.
        boundary_weights = [len(index_dict[boundary])/len(init_params)
                    for boundary in included_boundaries]
        weighted_gradients = [[grad * weight for grad in gradients_dict[boundary]]
                                for weight, boundary in zip(boundary_weights, included_boundaries)]
        total_gradients = [sum(grads) for grads in zip(*weighted_gradients)]
        all_loss = np.array([loss_dict[boundary] for boundary in included_boundaries])
        loss = np.sum(all_loss * np.array(boundary_weights))

        # Use the weighted gradients to train the policy network.
        self.session.run(self.train_policy, feed_dict={ph:grad for ph, grad in
                zip(self.gradients_ph_list, total_gradients)})

        # Reorganize states, actions, and rewards so they correspond to the same order
        # which they came in.
        # Python note: This is one of those unusual situations where we need to declare the
        # array first because we're populating it out of order.
        b0 = included_boundaries[0]
        state_shape = list(state_dict[b0].shape)
        state_shape[1] = len(init_params) # Adjust the shape to hold states for every initial
                                          # condition, not just the ones with this boundary.
        states = np.empty(state_shape, dtype=state_dict[b0].dtype)
        actions = []
        for action_part in action_dict[b0]:
            a_shape = list(action_part.shape)
            a_shape[1] = len(init_params)
            actions.append(np.empty(a_shape, dtype=action_part.dtype))
        rewards = []
        for reward_part in reward_dict[b0]:
            r_shape = list(reward_part.shape)
            r_shape[1] = len(init_params)
            rewards.append(np.empty(r_shape, dtype=reward_part.dtype))
        for boundary in included_boundaries:
            indexes = index_dict[boundary]
            states[:, indexes] = state_dict[boundary]
            for action_part, action_array in zip(action_dict[boundary], actions):
                action_array[:, indexes] = action_part
            for reward_part, reward_array in zip(reward_dict[boundary], rewards):
                reward_array[:, indexes] = reward_part
        actions = tuple(actions)
        rewards = tuple(rewards)

        extra_info = {}

        # rewards, actions, states are [timestep, initial_condition, ...]
        debug_mode = True
        training_plot_freq = self.log_freq * 5
        if debug_mode:
            #print("loss", loss)
            nans_in_grads = np.sum([np.sum(np.isnan(grad)) for grad in total_gradients])
            if nans_in_grads > 0:
                print("{} NaNs in grads".format(nans_in_grads))

            param_nans = self.session.run(self.params_nan_check)
            if any(param_nans.values()):
                print("NaNs detected in " +
                        str([name for name, is_nan in param_nans.items() if is_nan]))
            
            if not hasattr(self, 'iteration'):
                self.iteration = 0
            else:
                self.iteration += 1

            num_samples = 1
            ep_length = states.shape[0]
            num_inits = states.shape[1]
            ndims = len(states.shape[2:]) # Includes vector dimension.
            if self.env.vec_length == 1:
                ndims = ndims - 1

            #safe_state = states[np.logical_not(np.isnan(states).any(axis=
                                                                #tuple(range(ndims+2)[1:])))]
            #if len(safe_state) == 0:
                #raise Exception("All timesteps NaN. Stopping")

            random_reward_part = rewards[np.random.randint(len(rewards))]
            reward_spatial_dims = random_reward_part.shape[2:2+ndims]
            random_reward_indexes = tuple([slice(None),
                        np.random.randint(num_inits, size=num_samples)]
                        + [np.random.randint(nx, size=num_samples) for nx in reward_spatial_dims])
            sample_rewards = np.sum(random_reward_part[random_reward_indexes], axis=0)
            for i, reward in enumerate(sample_rewards):
                extra_info["sample_r"+str(i+1)] = reward

            random_action_part = actions[np.random.randint(len(actions))]
            action_spatial_dims = random_action_part.shape[2:2+ndims]
            random_action_indexes = tuple([np.random.randint(ep_length, size=num_samples),
                        np.random.randint(num_inits, size=num_samples)]
                        + [np.random.randint(nx, size=num_samples) for nx in action_spatial_dims])
            sample_actions = random_action_part[random_action_indexes]
            for i, action in enumerate(sample_actions):
                csv_string = '"'+(np.array2string(action, separator=',', precision=3)
                                    .replace('\n', '').replace(' ', '')) + '"'
                extra_info["sample_a"+str(i+1)] = csv_string

            if self.iteration % training_plot_freq == 0:
                for initial_condition_index in [0]: #range(states.shape[1]):
                    state_history = states[:, initial_condition_index]
                    suffix = "_train_iter{}_init{}".format(self.iteration, initial_condition_index)
                    self.env.plot_state_evolution(state_history=state_history,
                            no_true=True, suffix=suffix)

            action_nan = reward_nan = state_nan = False
            if any(np.isnan(action_part).any() for action_part in actions):
                print("NaN in actions during training")
                print("action shape", tuple(action_part.shape for actionpart in actions))
                # These won't work in ND.
                #print("timesteps with NaN:", np.isnan(actions).any(axis=(1,2,3,4)))
                #print("episodes with NaN:", np.isnan(actions).any(axis=(0,2,3,4)))
                #print("locations with NaN:", np.isnan(actions).any(axis=(0,1,3,4)))
                #print(actions[:27, 0, 65])
                action_nan = True
            if any(np.isnan(reward_part).any() for reward_part in rewards):
                print("NaN in rewards during training")
                #print("reward shape", rewards.shape)
                #print("timesteps with NaN:", np.isnan(rewards).any(axis=(1,2)))
                #print("episodes with NaN:", np.isnan(rewards).any(axis=(0,2)))
                #print("locations with NaN:", np.isnan(rewards).any(axis=(0,1)))
                reward_nan = True
            if np.isnan(states).any():
                print("NaN in states during training")
                #print("state shape", states.shape)
                #print("timesteps with NaN:", np.isnan(states).any(axis=(1,2)))
                #print("episodes with NaN:", np.isnan(states).any(axis=(0,2)))
                #print("locations with NaN:", np.isnan(states).any(axis=(0,1)))
                state_nan = True
            if (reward_nan or action_nan or state_nan):
                print("NaNs detected this round :(")
                raise Exception()
            #else:
                #print("No NaNs this round :)")

        output_info = {"loss":loss, "states": states, "actions":actions, "rewards":rewards}
        output_info.update(extra_info)
        return output_info

    def predict(self, state, deterministic=True):
        if not self._policy_ready or (not self._training_ready and not self.preloaded):
            raise Exception("No policy to predict with yet!")

        if not deterministic:
            print("Note: this model is strictly deterministic so using deterministic=False will"
            " have no effect.")

        single_obs_rank = len(self.env.observation_space.shape) - 1
        input_rank = len(state.shape)

        if input_rank == single_obs_rank:
            # Single local state.
            assert state.shape == self.env.observation_space.shape[1:]
            state = state[None]
            action = self.session.run(self.policy_output, feed_dict={self.policy_input_ph:state})
            action = action[0]

        elif input_rank == single_obs_rank + 1:
            # Batch of local states 
            # OR single global state.
            assert state.shape[1:] == self.env.observation_space.shape[1:]
            action = self.session.run(self.policy_output, feed_dict={self.policy_input_ph:state})

        elif input_rank == single_obs_rank + 2:
            # Batch of global states.
            assert state.shape[1:] == self.env.observation_space.shape
            batch_length = state.shape[0]
            spatial_length = state.shape[1]
            flattened_state = state.reshape((batch_length * spatial_length,) + state.shape[2:])
            flattened_action = self.session.run(self.policy_output,
                    feed_dict={self.policy_input_ph:flattened_state})
            action = flattened_action.reshape((batch_length, spatial_length,) + action.shape[1:])

        return action, None

    def save(self, path):
        """
        Save model paramaters to be loaded later.

        Based on StableBaselines save structure.
        """
        if not self._policy_ready or (not self._training_ready and not self.preloaded):
            raise Exception("No policy to save yet!")

        # I don't think we need to save any extra data? Loading requires loading the meta file
        # anyway, which contains all of the settings used here.
        extra_data = {}
        params = self.policy_params
        param_values = self.session.run(params)
        param_dict = OrderedDict((param.name, value) for param, value in zip(params, param_values))

        return save_to_zip(path, data=extra_data, params=param_dict)

    def load(self, path):
        if not self._loading_ready:
            self.setup_loading()

        data, param_dict = load_from_zip(path)
        if len(data) > 0:
            raise Exception("Miscellaneous data in GlobalBackpropModel was not empty."
                    " What did you put in there?")
        #self.__dict__.update(data) # Use this if we put something in extra_data in save().
                                    # (The keys need to be the same as the name of the field.)
        is_old_model = False
        feed_dict = {}
        for param in self.policy_params:
            placeholder = self.load_ph[param]
            try:
                if is_old_model:
                    name = param.name.replace("policy", "policy_net", 1)
                else:
                    name = param.name
                param_value = param_dict[name]
            except KeyError:
                if not is_old_model:
                    print("{} not in saved model.".format(param.name))
                    policy_net_name = param.name.replace("policy", "policy_net", 1)
                    if policy_net_name in param_dict:
                        print("Assuming older model that used \"policy_net\".")
                        param_value = param_dict[policy_net_name]
                        is_old_model = True
                    else:
                        raise Exception("\"{}\" is either corrupted or not a GlobalBackprop model."
                            .format(path))
                else:
                    print("{} not in saved model.".format(param.name))
                    raise Exception("\"{}\" is either corrupted or not a GlobalBackprop model."
                        .format(path))

            feed_dict[placeholder] = param_value

        self.session.run(self.load_op, feed_dict=feed_dict)
        print("Model loaded from {}".format(path))

        self.preloaded = True

class IntegrateRNN(Layer):
    def __init__(self, cell, name=None, dtype=tf.float64, swap_to_cpu=True):
        """
        Declare the RNN for PDE integration.

        Parameters
        ----------
        cell : tf.keras.layers.Layer
          The recurrent cell. Probably an IntegrateCell.
          Needs to have a call(previous_state) method returning (action, next_state).
        dtype : dtype
          Type of the state (probably tf.float32 or tf.float64).
        swap_to_cpu : bool
          Whether to copy Tensors to the CPU from the GPU. Used by the tf.while_loop() function.
          With long chains, too many Tensors on the GPU will exhaust the memory, but if the chain
          is relatively short (how short depends on the GPU), you can (probably) increase
          performance by keeping the Tensors on the GPU instead. True by default. Does nothing if
          only using the CPU anyway.
        """
        super().__init__(name=name, dtype=dtype)
        self.cell = cell
        self.swap_to_cpu = swap_to_cpu

    def build(self, input_size):
        super().build(input_size)

    def call(self, initial_state, num_steps):
        # - 2 for the batch and vector dims.
        spatial_dims = initial_state.get_shape().ndims - 2

        real_states_ta = tf.TensorArray(dtype=self.dtype, size=num_steps)
        #states_ta = tuple(tf.TensorArray(dtype=self.dtype, size=num_steps) for _ in
                #range(spatial_dims))
        actions_ta = tuple(tf.TensorArray(dtype=self.dtype, size=num_steps) for _ in
                range(spatial_dims))
        rewards_ta = tuple(tf.TensorArray(dtype=self.dtype, size=num_steps) for _ in
                range(spatial_dims))

        def condition_fn(time, *_):
            return time < num_steps

        def loop_fn(time, states_ta, actions_ta, rewards_ta, current_state):
            next_action, next_reward, next_state = self.cell(current_state)

            # Overwriting the TensorArrays is necessary to keep connections
            # through the entire network graph.
            states_ta = states_ta.write(time, next_state)
            #states_ta = tuple(sub_state_ta.write(time, sub_state) for
                    #sub_state_ta, sub_state in zip(states_ta, next_state))
            actions_ta = tuple(sub_action_ta.write(time, sub_action) for
                    sub_action_ta, sub_action in zip(actions_ta, next_action))
            rewards_ta = tuple(sub_reward_ta.write(time, sub_reward) for 
                    sub_reward_ta, sub_reward in zip(rewards_ta, next_reward))

            return (time + 1, states_ta, actions_ta, rewards_ta, next_state)

        initial_time = tf.constant(0, dtype=tf.int32)

        _time, real_states_ta, actions_ta, rewards_ta, final_state = tf.while_loop(
                cond=condition_fn,
                body=loop_fn,
                loop_vars=(initial_time, real_states_ta, actions_ta, rewards_ta, initial_state),
                parallel_iterations=10, # Iterations of the loop to run in parallel?
                                        # I'm not sure how that works.
                                        # 10 is the default value. Keras backend uses 32.
                swap_memory=self.swap_to_cpu,
                return_same_structure=True)

        real_states = real_states_ta.stack()
        #states = tuple(sub_state_ta.stack() for sub_state_ta in states_ta)
        actions = tuple(sub_action_ta.stack() for sub_action_ta in actions_ta)
        rewards = tuple(sub_reward_ta.stack() for sub_reward_ta in rewards_ta)
        return real_states, actions, rewards


class IntegrateCell(Layer):
    def __init__(self, prep_state_fn, policy_net, integrate_fn, reward_fn):
        super().__init__()

        self.prep_state_fn = prep_state_fn
        self.policy_net = policy_net
        self.integrate_fn = integrate_fn
        self.reward_fn = reward_fn

    def build(self, input_size):
        super().build(input_size)

    def call(self, state):
        # state has shape [batch, vector, location, ...]
        # (Batch shape may be unknown, other axes should be known.)

        real_state = state

        all_dims = real_state.get_shape().ndims
        # - 2 for the batch and vector dims.
        spatial_dims = all_dims - 2
        # outer_dims squeezes the vector dimension.
        if real_state.get_shape()[1] == 1:
            outer_dims = all_dims - 1
        else:
            outer_dims = all_dims

        # Use tf.map_fn to apply function across every element in the batch.
        tuple_type = (real_state.dtype,) * spatial_dims
        rl_state = tf.map_fn(self.prep_state_fn, real_state, dtype=tuple_type,
                                name='map_prep_state')

        rl_action = []
        for rl_state_part in rl_state:
            # The policy expects a batch so we don't need to use tf.map_fn;
            # however, it expects batches of INDIVIDUAL states, not every location at once, so we need
            # to combine the batch, vector, and location axes first (and then reshape the actions back).
            rl_state_shape = rl_state_part.shape.as_list()
            new_state_shape = [-1,] + rl_state_shape[outer_dims:]
            reshaped_state = tf.reshape(rl_state_part, new_state_shape)

            #print("original shape:", rl_state_shape)
            #print("new shape:", new_state_shape)

            # Future note: this works to apply a 1D agent along each dimension.
            # However, if we want an agent that makes use of a 2D stencil, or an agent that makes
            # use of multiple components of the vector at once, we'll need a different
            # implementation of IntegrateCell that somehow combines multiple stencils at each
            # location. (Or rather, something that can handle a prep_state function that does
            # that.)
            # This loop is equivalent to ExtendAgent2D.

            shaped_action = self.policy_net(reshaped_state)

            shaped_action_shape = shaped_action.shape.as_list()
            # Can't use rl_state_shape[:outer_dims] because rl_state_shape[0] is None; we need -1.
            rl_action_shape = [-1,] + rl_state_shape[1:outer_dims] + shaped_action_shape[1:]
            rl_action_part = tf.reshape(shaped_action, rl_action_shape)
            rl_action.append(rl_action_part)
        rl_action = tuple(rl_action)

        next_real_state = tf.map_fn(self.integrate_fn, (real_state, rl_state, rl_action),
                dtype=real_state.dtype, # Specify dtype because different from input (not a tuple).
                name='map_integrate')

        rl_reward = tf.map_fn(self.reward_fn, (real_state, rl_state, rl_action, next_real_state),
                dtype=tuple_type,
                name='map_reward')

        return rl_action, rl_reward, next_real_state
