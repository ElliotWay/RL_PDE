class Policy:
    """
    Interface for a policy.

    A policy need only have a predict method that returns an action based on a state.
    The predict method should take an optional deterministic parameter that makes the computation
    of the returned action deterministic.
    """
    def predict(self, state, deterministic=False):
        raise NotImplementedError
