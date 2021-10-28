import argparse

def positive_float(value):
    fvalue = float(value)
    if fvalue <= 0.0:
        raise argparse.ArgumentTypeError("{} is not positive".format(fvalue))
    return fvalue

def nonnegative_float(value):
    fvalue = float(value)
    if fvalue < 0.0:
        raise argparse.ArgumentTypeError("{} is not non-negative".format(fvalue))
    return fvalue

def positive_int(value):
    ivalue = int(value)
    if ivalue <= 0:
        raise argparse.ArgumentTypeError("{} is not positive".format(ivalue))
    return ivalue

def nonnegative_int(value):
    ivalue = int(value)
    if ivalue < 0:
        raise argparse.ArgumentTypeError("{} is not non-negative".format(ivalue))
    return ivalue

def proportion_float(value):
    fvalue = float(value)
    if fvalue < 0.0 or fvalue > 1.0:
        raise argparse.ArgumentTypeError("{} is not a proportion".format(fvalue))
    return fvalue

def float_dict(string_dict):
    pairs = string_dict.split(sep=',')
    # empty string returns empty dict
    if len(pairs) <= 1 and len(pairs[0]) == 0:
        return {}
    output_dict = {}
    for pair in pairs:
        match = re.fullmatch("([^=]+)=([^=]+)", pair)
        if not match:
            raise argparse.ArgumentTypeError("In \"{}\", \'{}\' must be key=value.".format(
                string_dict, pair))
        else:
            key = match.group(1)
            value = float(match.group(2))
            output_dict[key] = value
    return output_dict

# 3.8 has the 'extend' action, but we're in 3.7.
class ExtendAction(argparse.Action):
    def __init__(self, option_strings, dest, nargs=None, **kwargs):
        if nargs == None:
            nargs = '*'
        super().__init__(option_strings, dest, nargs=nargs, **kwargs)
    def __call__(self, parser, namespace, values, option_string=None):
        previous = getattr(namespace, self.dest)
        if previous is None:
            previous = []
        setattr(namespace, self.dest, previous + values)


