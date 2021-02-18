import os
import zipfile
import numpy as np

from scipy.special import softmax

from models import Model
from util.serialize import serialize_ndarray, deserialize_ndarray

class FixedOneStepModel(Model):
    """
    A Model that aims to learn an optimal set of weights to apply at every location.
    """
    def __init__(self, env, args):
        #self.weights = np.zeros(env.action_space.shape)
        self.weights = np.random.uniform(-1.0, 1.0, env.action_space.shape)
        self.learning_rate = args.learning_rate
        self.learning_rate_decay = 1.0 #0.999
        self.action_noise = 0.1
        self.noise_decay = 1.0 #0.99
    
    def train(self, s, a, r, s2, done):
        # Not a conventional means of training, this, given we're trying to learn continuous
        # parameters directly.
        actions = np.array(a)
        rewards = np.array(r)

        previous_weights = self.weights.copy()

        # Some tricky reshaping to multiply by the rewards over the batch dimension.
        rewards = rewards.reshape([-1] + ([1] * len(self.weights.shape)))

        #reward_weights = softmax(rewards)
        #self.weights = ((1 - self.learning_rate) * self.weights
                #+ self.learning_rate * np.sum(actions * reward_weights, axis=0))

        deltas = actions - self.weights
        normalized_rewards = softmax(rewards)
        total_grad = np.sum(deltas * normalized_rewards, axis=0)
        self.weights += self.learning_rate * total_grad

        normalized_weights = np.concatenate((
                softmax(self.weights[:len(self.weights)//2]),
                softmax(self.weights[len(self.weights)//2:])))

        info_dict = {'current_lr': self.learning_rate,
                     'current_action_noise': self.action_noise,
                     'total_change': np.sum(np.abs(self.weights - previous_weights)),
                     'weights': ",".join(["{:0.3f}".format(x) for x in self.weights]),
                     'normalized_weights': ",".join(["{:0.3f}".format(x) for x in
                         normalized_weights]),
                     }

        self.learning_rate *= self.learning_rate_decay
        self.action_noise *= self.noise_decay

        return info_dict

    def predict(self, state, deterministic=True):
        output_shape = (len(state),) + self.weights.shape
        actions = np.broadcast_to(self.weights, output_shape)
        if deterministic:
            return actions, None
        else:
            noisy_actions = actions + np.random.uniform(
                    -self.action_noise, self.action_noise, actions.shape)
            return noisy_actions, None

    def save(self, path):
        _, ext = os.path.splitext(path)
        if ext == "":
            path += ".zip"

        with zipfile.ZipFile(path, "w") as file_:
            file_.writestr("weights", serialize_ndarray(self.weights))
            file_.writestr("lr", str(self.learning_rate))
            file_.writestr("lr_decay", str(self.learning_rate_decay))
            file_.writestr("noise", str(self.action_noise))
            file_.writestr("noise_decay", str(self.noise_decay))

        return path

    def load(self, path):
        with zipfile.ZipFile(path, "r") as file_:
            self.weights = deserialize_ndarray(file_.read("weights"))
            self.learning_rate = float(file_.read("lr"))
            self.learning_rate_decay = float(file_.read("lr_decay"))
            self.action_noise = float(file_.read("noise"))
            self.noise_decay = float(file_.read("noise_decay"))
