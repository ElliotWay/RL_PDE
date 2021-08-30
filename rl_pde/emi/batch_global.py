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

    # Declare this lazily so it doesn't need to be declared during testing.
    def _declare_solution_env(self):
        if self.weno_solution_env is None:
            # To record the "full" error, that is, the difference between the solution following the
            # agent and the solution following WENO from the beginning, we need to keep a separate copy
            # of the WENO solution.
            args_copy = Namespace(**vars(self.args))
            if args_copy.reward_mode is not None:
                args_copy.reward_mode = args_copy.reward_mode.replace('one-step', 'full')
            if not "random" in args_copy.init_type and args_copy.fixed_timesteps:
                args_copy.memoize = True # Memoizing means we don't waste time and memory recomputing
                                         # the solution each time.
            else:
                args_copy.memoize = False # Unfortunately we HAVE to recompute each time if the
                                          # environment is randomized.
            args_copy.analytical = False

            self.weno_solution_env = build_env(args_copy.env, args_copy)

    def training_episode(self, env):
        self._declare_solution_env()

        num_inits = self.args.batch_size
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

        extra_info = self._model.train(initial_conditions, init_params)

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
        info_dict['timesteps'] = num_inits * self.args.ep_length
        info_dict.update(extra_info)
        return info_dict

    def get_policy(self):
        return self.policy
