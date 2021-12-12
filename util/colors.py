# This file is a standard list of colors. Multiple files can refer to these so colors are
# consistent across a variety of plots.
import matplotlib.colors as colors

WENO_COLOR = 'tab:blue'
WENO_KWARGS = {'color':WENO_COLOR, 'linestyle':'', 'marker':'x'}
RL_COLOR = 'tab:orange'
RL_KWARGS = {'color':RL_COLOR, 'linestyle':'', 'marker':'.'}
ANALYTICAL_COLOR = 'black' #'tab:pink'
ANALYTICAL_KWARGS = {'color':ANALYTICAL_COLOR, 'linestyle':'-', 'marker':'', 'linewidth':0.75}

def get_agent_kwargs(filename, label, just_color=False):
    """
    Infer whether the agent is RL, WENO, or analytical based on the filename
    and label, then return the style kwargs for that agent.
    If the filename and label contain no information, assume the file is an RL agent.
    The returned dict is a copy and can be safely modified.
    """
    if 'weno' in filename.lower() or 'weno' in label.lower():
        if just_color:
            return {'color':WENO_COLOR}
        else:
            return dict(WENO_KWARGS) # Return a copy of the dict so the caller can change it.
    elif (any(name in filename.lower() for name in ['analytical', 'true'])
            or any(name in label.lower() for name in ['analytical', 'true'])):
        if just_color:
            return {'color':ANALYTICAL_COLOR}
        else:
            return dict(ANALYTICAL_KWARGS)
    else:
        if not (any(name in filename.lower() for name in ['rl', 'agent'])
                or any(name in lable.lower() for name in ['rl', 'agent'])):
            print(f"Warning: can't determine type of {filename};"
                    + " assuming it is from an RL agent.")
        if just_color:
            return {'color':RL_COLOR}
        else:
            return dict(RL_KWARGS)

TRAIN_COLOR = 'tab:gray' #'black'
# avg eval color is the same as RL color because they both refer to the agent in test environments.
AVG_EVAL_COLOR = 'tab:orange'
EVAL_ENV_COLORS = ['b', 'r', 'g', 'm', 'c', 'y']

WENO_ORDER_COLORS = [None, None, 'g', 'b', 'r', 'y', 'c', 'm']
# Polynomial colors line up so that WENO color i is poly color 2i-1.
# Even-indexed polynomial colors are the appropriate colors between the odd-indexed colors.
POLY_COLORS = [None, 'grey', (0.4, 0.7, 0.0), 'g', 'c', 'b', 'm', 'r',
        'tab:orange', 'y', (0.375, 0.75, 0.375), 'c', (0.375, 0.375, 0.75), 'm']

ALTERNATE_WEIGHTS = [1.0, 0.7, 1/0.7, 0.4, 1/0.4, 0.1, 1/0.1]
PERMUTATIONS = [lambda c, index=index:permute_color(c, index) for index in range(len(ALTERNATE_WEIGHTS))]

def permute_color(color, index):
    r, g, b, alpha = colors.to_rgba(color)

    h, s, v = colors.rgb_to_hsv([r, g, b])

    # Handle base colors close to black differently.
    if v < 0.2:
        new_v = 1.0 - (1.0 - v) / (index + 1)
    else:
        new_v = max(0.0, min(1.0, v * (1.0 / ALTERNATE_WEIGHTS[index])))
    new_s = max(0.0, min(1.0, s * ALTERNATE_WEIGHTS[index]))
    
    new_r, new_g, new_b = colors.hsv_to_rgb([h, new_s, new_v])

    return [new_r, new_g, new_b, alpha]
