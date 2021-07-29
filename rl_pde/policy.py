class Policy:
    """
    Interface for a policy.

    A policy need only have a predict method that returns an action based on a state.
    The predict method should take an optional deterministic parameter that makes the computation
    of the returned action deterministic.

    Returns action, None
    (The None is a holdover from a system that could handle recurrent networks - it should be
    the recurrent state if there is one.)
    """
    def predict(self, state, deterministic=False):
        raise NotImplementedError
