
from models import SACModel, PolicyGradientModel, TestModel
from models.full import GlobalBackpropModel
from models.fixed import FixedOneStepModel

from rl_pde.emi import BatchEMI, HomogenousMARL_EMI, BatchGlobalEMI, StandardEMI, TestEMI

# Various functions to convert strings into something else.
# Similar to the motivation behind function_dict.py, it's easier to have a string as an argument
# and to serialize a string, than to use arbitrary objects. To avoid repeated code, these functions
# give the associations between strings and the associated object.

# TODO: A better alternative to this file might be a "create arg parser" file.
# That could create the arguments for run_train and run_test while having the associations of
# argument values in the same file.

# TODO: There are more of these that could make run_train and run_test cleaner. Something to do
# later.

def get_model_class(model_name):
    if model_name == 'sac':
        return SACModel
    elif model_name == 'full':
        return GlobalBackpropModel
    elif model_name == 'pg' or model_name == 'reinforce':
        return PolicyGradientModel
    elif model_name == 'test':
        return TestModel
    elif model_name == "fixed-1step" or model_name == "fixed":
        return FixedOneStepModel
    else:
        raise Exception("Unrecognized model type: \"{}\"".format(model_name))

def get_emi_class(emi_name):
    if emi_name == 'batch':
        return BatchEMI
    elif emi_name == 'marl':
        return HomogenousMARL_EMI
    elif emi_name == 'batch-global':
        return BatchGlobalEMI
    elif emi_name == 'std' or emi_name == 'standard':
        return StandardEMI
    elif emi_name == 'test':
        return TestEMI
    else:
        raise Exception("Unrecognized EMI: \"{}\"".format(emi_name))
