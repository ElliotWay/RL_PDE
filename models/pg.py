import os
import zipfile
import numpy as np

from models import Model
from util.misc import serialize_ndarray, deserialize_ndarray

class PolicyGradientModel(Model):
    def __init__(self, env, args):
        pass
    
    def train(self, s, a, r, s2, done):
        
        pass

    def predict(self, state, deterministic=True):
        pass

    def save(self, path):

        return path

    def load(self, path):
        pass
