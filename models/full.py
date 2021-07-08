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
        Note that as of writing this, the env needs to be specifically a WENOBurgersEnv - a general
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

            cell = IntegrateCell(
                    prep_state_fn=self.env.tf_prep_state,
                    policy_net=self.wrapped_policy,
                    integrate_fn=self.env.tf_integrate,
                    reward_fn=self.env.tf_calculate_reward,
                    )
            rnn = IntegrateRNN(cell)

            # initial_state_ph is the REAL physical initial state
            # (It should not contain ghost cells - those should be handled by the prep_state function
            # that converts real state to rl state.)
            self.initial_state_ph = tf.placeholder(dtype=self.dtype,
                    shape=(None, self.args.nx), name="init_real_state")

            #TODO possibly restrict num_steps to something smaller?
            # We'd then need to sample timesteps from a WENO trajectory.
            #NOTE 4/14 - apparently it works okay with the full episode! We may still want to restrict
            # this if resource constraints become an issue with more complex environments.

            states, actions, rewards = rnn(self.initial_state_ph, num_steps=self.args.ep_length)
            self.states = states
            self.actions = actions
            self.rewards = rewards

            assert len(self.rewards.shape) == 3
            # Sum over the timesteps in the trajectory (axis 0),
            # then average over each location and the batch (axes 2 and 1).
            self.loss = -tf.reduce_mean(tf.reduce_sum(rewards, axis=0))

            gradients = tf.gradients(self.loss, self.policy_params)
            self.grads = list(zip(gradients, self.policy_params))

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
                placeholder = tf.placeholder(dtype=param.dtype, shape=param.shape)
                self.load_op[param] = param.assign(placeholder)
                self.load_ph[param] = placeholder

            self._loading_ready = True

    def train(self, initial_state):
        if not self._training_ready:
            self.setup_training()

        feed_dict = {self.initial_state_ph:initial_state}

        _, grads, loss, rewards, actions, states = self.session.run(
                [self.train_policy, self.grads, self.loss, self.rewards, self.actions, self.states],
                feed_dict=feed_dict)

        extra_info = {}

        # rewards, actions, states are [timestep, initial_condition, ...]

        debug_mode = True
        training_plot_freq = self.log_freq * 5
        if debug_mode:
            #print("loss", loss)
            nans_in_grads = np.sum([np.sum(np.isnan(grad)) for grad in grads])
            if nans_in_grads > 0:
                print("{} NaNs in grads".format(nans_in_grads))

            # The way session.run can handle dicts is convenient.
            param_nans = self.session.run(self.params_nan_check)
            if any(param_nans.values()):
                print("NaNs detected in " +
                        str([name for name, is_nan in param_nans.items() if is_nan]))
            
            if not hasattr(self, 'iteration'):
                self.iteration = 0
            else:
                self.iteration += 1

            safe_state = states[np.logical_not(np.isnan(states).any(axis=(1,2)))]
            #if len(safe_state) == 0:
                #raise Exception("All timesteps NaN. Stopping")

            num_samples = 1
            ep_length = actions.shape[0]
            num_inits = actions.shape[1]
            spatial_width = actions.shape[2]
            sample_rewards = np.sum(rewards[:, np.random.randint(num_inits, size=num_samples),
                np.random.randint(spatial_width, size=num_samples)], axis=0)
            for i, reward in enumerate(sample_rewards):
                extra_info["sample_r"+str(i+1)] = reward
            sample_actions = actions[np.random.randint(ep_length, size=num_samples),
                    np.random.randint(num_inits, size=num_samples),
                    np.random.randint(spatial_width, size=num_samples)]
            for i, action in enumerate(sample_actions):
                csv_string = '"'+(np.array2string(action, separator=',', precision=3)
                                    .replace('\n', '').replace(' ', '')) + '"'
                extra_info["sample_a"+str(i+1)] = csv_string

            if self.iteration % training_plot_freq == 0:
                for initial_condition_index in [0]:#range(states.shape[1]):
                    state_history = states[:, initial_condition_index]
                    suffix = "_train_iter{}_init{}".format(self.iteration, initial_condition_index)
                    self.env.plot_state_evolution(state_history=state_history,
                            no_true=True, suffix=suffix)

            if np.isnan(actions).any():
                print("NaN in actions during training")
                print("action shape", actions.shape)
                print("timesteps with NaN:", np.isnan(actions).any(axis=(1,2,3,4)))
                #print("episodes with NaN:", np.isnan(actions).any(axis=(0,2,3,4)))
                #print("locations with NaN:", np.isnan(actions).any(axis=(0,1,3,4)))
                print(actions[:27, 0, 65])
            if np.isnan(rewards).any():
                print("NaN in rewards during training")
                #print("reward shape", rewards.shape)
                #print("timesteps with NaN:", np.isnan(rewards).any(axis=(1,2)))
                #print("episodes with NaN:", np.isnan(rewards).any(axis=(0,2)))
                #print("locations with NaN:", np.isnan(rewards).any(axis=(0,1)))
            if np.isnan(states).any():
                print("NaN in states during training")
                #print("state shape", states.shape)
                #print("timesteps with NaN:", np.isnan(states).any(axis=(1,2)))
                #print("episodes with NaN:", np.isnan(states).any(axis=(0,2)))
                #print("locations with NaN:", np.isnan(states).any(axis=(0,1)))
            if np.isnan(states).any() or np.isnan(actions).any() or np.isnan(states).any():
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

        feed_dict = {}
        for param in self.policy_params:
            placeholder = self.load_ph[param]
            try:
                param_value = param_dict[param.name]
            except KeyError:
                print("{} not in saved model.".format(param.name))
                policy_net_name = param.name.replace("policy", "policy_net", 1)
                if policy_net_name in param_dict:
                    print("Assuming older model that used \"policy_net\".")
                    param_value = param_dict[policy_net_name]
                else:
                    raise Exception("\"{}\" is either corrupted or not a GlobalBackprop model."
                        .format(path))

            feed_dict[placeholder] = param_value

        self.session.run(self.load_op, feed_dict=feed_dict)
        print("Model loaded from {}".format(path))

        self.preloaded = True

class IntegrateRNN(Layer):
    def __init__(self, cell, dtype=tf.float64, swap_to_cpu=True):
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
        super().__init__(dtype=dtype)
        self.cell = cell
        self.swap_to_cpu = swap_to_cpu

    def build(self, input_size):
        super().build(input_size)

    def call(self, initial_state, num_steps):
        states_ta = tf.TensorArray(dtype=self.dtype, size=num_steps)
        actions_ta = tf.TensorArray(dtype=self.dtype, size=num_steps)
        rewards_ta = tf.TensorArray(dtype=self.dtype, size=num_steps)

        def condition_fn(time, *_):
            return time < num_steps

        def loop_fn(time, states_ta, actions_ta, rewards_ta, current_state):
            next_action, next_reward, next_state = self.cell(current_state)

            # Overwriting the TensorArrays is necessary to keep connections
            # through the entire network graph.
            states_ta = states_ta.write(time, next_state)
            actions_ta = actions_ta.write(time, next_action)
            rewards_ta = rewards_ta.write(time, next_reward)

            return (time + 1, states_ta, actions_ta, rewards_ta, next_state)

        initial_time = tf.constant(0, dtype=tf.int32)

        _time, states_ta, actions_ta, rewards_ta, final_state = tf.while_loop(
                cond=condition_fn,
                body=loop_fn,
                loop_vars=(initial_time, states_ta, actions_ta, rewards_ta, initial_state),
                parallel_iterations=10, # Iterations of the loop to run in parallel?
                                        # I'm not sure how that works.
                                        # 10 is the default value. Keras backend uses 32.
                swap_memory=self.swap_to_cpu,
                return_same_structure=True)

        states = states_ta.stack()
        actions = actions_ta.stack()
        rewards = rewards_ta.stack()
        return states, actions, rewards


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
        # state has shape [batch, location, ...]
        # (Batch shape may be unknown, other axes should be known.)

        real_state = state
        # Use tf.map_fn to apply function across every element in the batch.
        rl_state = tf.map_fn(self.prep_state_fn, real_state, name='map_prep_state')

        # The policy expects a batch so we don't need to use tf.map_fn;
        # however, it expects batches of INDIVIDUAL states, not every location at once, so we need
        # to combine the batch and location axes first (and then reshape the actions back).
        rl_state_shape = rl_state.shape.as_list()
        new_state_shape = [-1,] + rl_state_shape[2:]
        reshaped_state = tf.reshape(rl_state, new_state_shape)

        shaped_action = self.policy_net(reshaped_state)

        shaped_action_shape = shaped_action.shape.as_list()
        rl_action_shape = [-1, rl_state_shape[1]] + shaped_action_shape[1:]
        rl_action = tf.reshape(shaped_action, rl_action_shape)

        next_real_state = tf.map_fn(self.integrate_fn, (real_state, rl_state, rl_action),
                dtype=real_state.dtype, # Specify dtype because different from input (not a tuple).
                name='map_integrate')

        rl_reward = tf.map_fn(self.reward_fn, (real_state, rl_state, rl_action, next_real_state),
                dtype=real_state.dtype,
                name='map_reward')

        return rl_action, rl_reward, next_real_state
