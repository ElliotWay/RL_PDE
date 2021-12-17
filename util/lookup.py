import tensorflow as tf

#import models.model.TestModel as TestModel
#import models.full.GlobalBackpropModel as GlobalBackpropModel
#import models.sac.sac_wrapper as SACModel
#import models.pg.PolicyGradientModel as PolicyGradientModel
#import models.fixed.FixedOneStepModel as FixedOneStepModel
import models

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
    if model_name == 'full':
        return models.full.GlobalBackpropModel
    elif model_name == 'sac':
        return models.SACModel
    elif model_name == 'pg': # Policy Gradient
        return models.PolicyGradientModel
    elif model_name == "fixed-1step":
        return models.fixed.FixedOneStepModel
    elif model_name == 'test':
        return TestModel
    elif model_name == "ddpg":
        raise Exception("Some DDPG files still exist in this repo, but they are not currently"
                + " supported.")
    else:
        raise Exception("Unrecognized model type: \"{}\"".format(model_name))

def get_model_dims(model_name):
    # Currently, they're ALL 1-dimensionsal.
    # We'll need to change this later when 2-dimensional models exist.
    # Note that this refers to the dimensions of the underlying policy, i.e. the stencil that the
    # policy operates on.
    # We only need to worry about this when we have a policy using a multi-dimensional stencil.
    return 1

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

def get_activation(activation_name):
    if activation_name == 'relu':
        return tf.nn.relu
    elif activation_name == 'sigmoid':
        return tf.nn.sigmoid
    elif activation_name == 'tanh':
        return tf.nn.tanh

