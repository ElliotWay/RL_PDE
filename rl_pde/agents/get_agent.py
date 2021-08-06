from rl_pde.agents.weno_agents import StandardWENOAgent, StandardWENO2DAgent
from rl_pde.agents.basic_agents import StationaryAgent, EqualAgent, MiddleAgent, LeftAgent, RightAgent
from rl_pde.agents.basic_agents import RandomAgent
from rl_pde.agents.extend_agent_2d import ExtendAgent2D

def get_agent(name, order, mode="weno", dimensions=1):
    if dimensions > 2:
        raise NotImplementedError()

    if (name == "default" or name == "none"
        or name == "weno" or name == "std"):
        if dimensions == 1:
            return StandardWENOAgent(order=order, mode=mode)
        elif dimensions == 2:
            return StandardWENO2DAgent(order=order, mode=mode)
            # This would be work but would be substantially slower:
            #return ExtendAgent2D(StandardWENOAgent(order=order, mode=mode))

    else:
        if name == "stationary":
            agent = StationaryAgent(order=args.order, mode=mode)
        elif name == "equal":
            agent = EqualAgent(order=args.order)
        elif name == "middle":
            agent = MiddleAgent(order=args.order)
        elif name == "left":
            agent = LeftAgent(order=args.order)
        elif name == "right":
            agent = RightAgent(order=args.order)
        elif name == "random":
            agent = RandomAgent(order=args.order, mode=mode)
        else:
            return None

        if dimensions == 2:
            return ExtendAgent2D(agent)
        elif dimensions == 1:
            return agent
 