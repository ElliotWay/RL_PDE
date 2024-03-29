import matplotlib as mpl
# This file is a standard list of colors. Multiple files can refer to these so colors are
# consistent across a variety of plots.
import matplotlib.colors as colors

INIT_COLOR = 'black'
INIT_KWARGS = {'color':INIT_COLOR, 'linestyle':'--'}
WENO_COLOR = 'tab:blue'
WENO_KWARGS = {'color':WENO_COLOR, 'linestyle':'', 'marker':'x',
        'markersize':1.5*mpl.rcParams['lines.markersize']}
# Second WENO kwargs for WENO that should be similar but different.
WENO_COLOR2 = [0.51, 0.68, 0.80]
# Intentionally use the original WENO color if using WENO_KWARGS2, as the markers will distinguish
# the difference.
WENO_KWARGS2 = {'color':WENO_COLOR, 'linestyle':'', 'marker':'+',
        'markersize':2.25*mpl.rcParams['lines.markersize']}
RL_COLOR = [1.0, 0.67, 0.25]#'tab:orange'
RL_KWARGS = {'color':RL_COLOR, 'linestyle':'', 'marker':'o'}
# Second RL kwargs for RL that should be similar but different.
#RL_COLOR2 = [0.95, 0.70, 0.48]
#RL_COLOR2 = [1.0, 0.67, 0.25]
#RL_COLOR2 = [0.8, 0.8, 0.4]
#RL_COLOR2 = [154/255, 113/255, 151/255]
#RL_COLOR2 = [168/255, 87/255, 126/255]
RL_COLOR2 = [143/255, 37/255, 12/255]

RL_KWARGS2= {'color':RL_COLOR2, 'linestyle':'', 'marker':'o'}
ANALYTICAL_COLOR = 'black' #'tab:pink'
ANALYTICAL_KWARGS = {'color':ANALYTICAL_COLOR, 'linestyle':'-', 'marker':'', 'linewidth':0.75}

agent_count = 0

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
                or any(name in label.lower() for name in ['rl', 'agent'])):
            print(f"Warning: can't determine type of {filename};"
                    + " assuming it is from an RL agent.")
        global agent_count
        agent_count += 1
        if agent_count == 1:
            if just_color:
                return {'color':RL_COLOR}
            else:
                return dict(RL_KWARGS)
        elif agent_count == 2:
            if just_color:
                return {'color':RL_COLOR2}
            else:
                return dict(RL_KWARGS2)

TRAIN_COLOR = [0.25,0.25,0.25]#'dimgray'#'tab:gray' #'black'
TRAIN_COLORS = [TRAIN_COLOR, [99/255, 163/255, 117/255], [56/255,91/255,181/255]]
# avg eval color is the same as RL color because they both refer to the agent in test environments.
AVG_EVAL_COLOR = RL_COLOR#'tab:orange'
AVG_EVAL_COLORS = [RL_COLOR, RL_COLOR2, [172/255,60/255,189/255]]
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
