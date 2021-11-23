# This file is a standard list of colors. Multiple files can refer to these so colors are
# consistent across a variety of plots.

TRAIN_COLOR = 'black'
AVG_EVAL_COLOR = 'tab:orange'
EVAL_ENV_COLORS = ['b', 'r', 'g', 'm', 'c', 'y']

WENO_ORDER_COLORS = [None, None, 'g', 'b', 'r', 'y', 'c', 'm']
# Polynomial colors line up so that WENO color i is poly color 2i-1.
# Even-indexed polynomial colors are the appropriate colors between the odd-indexed colors.
POLY_COLORS = [None, 'grey', (0.4, 0.7, 0.0), 'g', 'c', 'b', 'm', 'r',
        'tab:orange', 'y', (0.375, 0.75, 0.375), 'c', (0.375, 0.375, 0.75), 'm']
