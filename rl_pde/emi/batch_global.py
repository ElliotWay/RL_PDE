import numpy as np
from argparse import Namespace

from envs import build_env
from rl_pde.run import rollout
from rl_pde.emi.emi import EMI, PolicyWrapper
from rl_pde.emi.emi import OneDimensionalStencil
from rl_pde.emi.batch import UnbatchedEnvPL, UnbatchedPolicy

class BatchGlobalEMI(EMI, OneDimensionalStencil):
    """
    EMI that breaks samples along the first dimension (the physical dimension). However, unlike
    BatchEMI, BatchGlobalEMI is built to handle a GlobalModel. The GlobalModel needs information
    about the underlying environment, so the model itself sees the original environment, but the
    policy returned by get_policy() acts identically to the policy returned by get_policy() in
    BatchEMI.

    Notably, the "training episode" trains with a set of initial conditions, instead of with a set
    of trajectories.
    """
    def __init__(self, env, model_cls, args, action_adjust=None, obs_adjust=None):
        # In the global model, the model sees the entire environment anyway, so we need to give it
        # the original unmodified environment. (The model gets information about the adjust
        # functions in the args dict.)
        #TODO This is inelegant. Can the model be constructed in a way that it is unaware of this
        # information?
        self._model = model_cls(env, args)

        # The external policy still sees the same interface, so we still use the same decorators as
        # with BatchEMI. Note that unlike in BatchEMI, self.policy is NOT used during training.
        unbatched_env = UnbatchedEnvPL(env, flatten=False)
        unbatched_policy = UnbatchedPolicy(env, unbatched_env, self._model)
        self.policy = PolicyWrapper(unbatched_policy, action_adjust, obs_adjust)

        self.args = args # Not ideal. This EMI has too much visibility if it keeps the whole args.

        self.weno_solution_env = None
        self.analytical_solution_env = None
        self.solution_env_states = None

    # Declare this lazily so it doesn't need to be declared during testing.
    def _declare_solution_env(self):
        if self.weno_solution_env is None:
            # To record the "full" error, that is, the difference between the solution following the
            # agent and the solution following WENO from the beginning, we need to keep a separate copy
            # of the WENO solution.
            env_args_copy = Namespace(**vars(self.args.e))
            if env_args_copy.reward_mode is not None:
                env_args_copy.reward_mode = env_args_copy.reward_mode.replace('one-step', 'full')
            # Note: just use the given memoization parameter for now.
            # The random environments have been discretized so that they can be memoized anyway.
            #if not "random" in args_copy.init_type and args_copy.fixed_timesteps:
                #args_copy.memoize = True # Memoizing means we don't waste time and memory recomputing
                                         # the solution each time.
            #else:
                #args_copy.memoize = False # Unfortunately we HAVE to recompute each time if the
                                          # environment is randomized.
            env_args_copy.analytical = False

            self.weno_solution_env = build_env(self.args.env, env_args_copy)

        if self.analytical_solution_env is None:
            env_args_copy = Namespace(**vars(self.args.e))
            if env_args_copy.reward_mode is not None:
                env_args_copy.reward_mode = env_args_copy.reward_mode.replace('one-step', 'full')
            env_args_copy.analytical = True
            self.analytical_solution_env = build_env(self.args.env, env_args_copy)

    # Declare this lazily so it doesn't need to be declared during testing.
    def _generate_solution_env_states(self, all_init_params, analytical=True, last_only=False):
        """
        Generate solution env states for calculating rewards during training.

        Parameters
        ---------
        analytical: bool
            If true, use true analytical solution, else use full WENO solution.
        last_only: bool
            If true, generate only last solution state for calculating rewards, else generate all steps.
        """

        if self.solution_env_states is None:
            solution_env_states = []
            if analytical:
                for init_params in all_init_params:
                    # Using init_params instead of copying the state directly allows the solution to use
                    # memoization.
                    self.analytical_solution_env.init_params = init_params
                    self.analytical_solution_env.reset()

                    # env.evolve() evolves the state using the internal solution (WENO in this case).
                    self.analytical_solution_env.evolve()

                    if last_only:
                        solution_env_states.append(self.analytical_solution_env.state_history[-1]
                                                   [self.analytical_solution_env.grid.real_slice])
                        self.solution_env_states = np.array(solution_env_states)
                    else:
                        solution_env_states.append([item[self.analytical_solution_env.grid.real_slice]
                                                    for item in self.analytical_solution_env.state_history])
                        self.solution_env_states = np.swapaxes(np.array(solution_env_states), 0, 1)

            else:
                for init_params in all_init_params:
                    # Using init_params instead of copying the state directly allows the solution to use
                    # memoization.
                    self.weno_solution_env.init_params = init_params
                    self.weno_solution_env.reset()

                    # env.evolve() evolves the state using the internal solution (WENO in this case).
                    self.weno_solution_env.evolve()

                    if last_only:
                        solution_env_states.append(self.weno_solution_env.state_history[-1]
                                                   [self.weno_solution_env.grid.real_slice])
                        self.solution_env_states = np.array(solution_env_states)
                    else:
                        solution_env_states.append([item[self.weno_solution_env.grid.real_slice]
                                                    for item in self.weno_solution_env.state_history])
                        self.solution_env_states = np.swapaxes(np.array(solution_env_states), 0, 1)

        return self.solution_env_states

    def training_episode(self, env):
        self._declare_solution_env()

        num_inits = self.args.m.batch_size
        initial_conditions = []
        init_params = []
        for _ in range(num_inits):
            rl_state = env.reset()
            # Important to use copy - grid.get_real returns the writable grid cells, which are
            # changed by the environment.
            phys_state = np.copy(env.grid.get_real())
            initial_conditions.append(phys_state)
            init_params.append(env.grid.init_params) # This is starting to smell.
        initial_conditions = np.array(initial_conditions)

        solution_states = self._generate_solution_env_states(init_params)
        extra_info = self._model.train(initial_conditions, init_params, solution_states)

        states = extra_info['states']
        del extra_info['states']
        actions = extra_info['actions']
        del extra_info['actions']
        rewards = extra_info['rewards']
        del extra_info['rewards']

        # Compute the L2 error of the final state with the final state of the WENO solution.
        l2_errors = []
        for init_params, final_state in zip(init_params, states[-1]):
            # Using init_params instead of copying the state directly allows the solution to use
            # memoization.
            self.weno_solution_env.init_params = init_params
            self.weno_solution_env.reset()
            
            # env.evolve() evolves the state using the internal solution (WENO in this case).
            self.weno_solution_env.evolve()
            # Note that it effectively has 2 copies: the state and the solution state this means
            # when we overwrite the state with the state from the agent, it has still has the
            # solution copy with which to compute the error.
            self.weno_solution_env.force_state(final_state)
            l2_errors.append(self.weno_solution_env.compute_l2_error())
        avg_l2_error = np.mean(l2_errors)

        info_dict = {}
        # Note that information coming from the model
        # has dimensions [timestep, initial_condition, ...], so reducing across time is reducing
        # across axis 0.
        info_dict['reward'] = tuple([np.mean(np.sum(reward_part, axis=0), axis=0) for
                                        reward_part in rewards])
        info_dict['l2_error'] = avg_l2_error
        info_dict['timesteps'] = num_inits * self.args.e.ep_length
        info_dict.update(extra_info)
        return info_dict

    def get_policy(self):
        return self.policy
