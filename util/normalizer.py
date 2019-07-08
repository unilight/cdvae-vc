
import tensorflow as tf
import numpy as np

class MinMaxScaler(object):
    """ Normalizing x to [-1, 1]
    """
    def __init__(self, xmin, xmax):
        self.xmin = xmin
        self.xmax = xmax
        self.xscale = xmax - xmin
    
    def forward_process(self, x):
        x = (x - self.xmin) / self.xscale
        return tf.clip_by_value(x, 0., 1.) * 2. - 1.
    
    def forward_process_np(self, x):
        x = (x - self.xmin) / self.xscale
        return x * 2. - 1.

    def backward_process(self, x, low=None, high=None):
        if (low is None) and (high is None):
            return (x * .5 + .5) * self.xscale + self.xmin
        else:
            return (x * .5 + .5) * self.xscale[low:high] + self.xmin[low:high]
    
class StandardScaler(object):
    """ Normalizing x to zero mean, unit variance
    """
    def __init__(self, mu, std):
        self.mu = mu
        self.std = std
    
    def forward_process(self, x):
        return (x - self.mu) / (self.std + EPSILON)
    
    def backward_process(self, x):
        return x * self.std + self.mu 
