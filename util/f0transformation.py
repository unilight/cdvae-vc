import numpy as np
import tensorflow as tf
        
def log_linear_transformation(f0, stats, module='numpy'):
    """ linear transformation of log-f0
        either in `numpy` or `tf`
    """
    if module=='numpy':
        lf0 = np.where(f0 > 1., np.log(f0), f0)
        lf0 = np.where(lf0 > 1., (lf0 - stats['mu_s'])/stats['std_s'] * stats['std_t'] + stats['mu_t'], lf0)
        lf0 = np.where(lf0 > 1., np.exp(lf0), lf0)
        return lf0
    elif module=='tf':
        lf0 = tf.where(f0 > 1., tf.log(f0), f0)
        lf0 = np.where(lf0 > 1., (lf0 - stats['mu_s'])/stats['std_s'] * stats['std_t'] + stats['mu_t'], lf0)
        lf0 = tf.where(lf0 > 1., tf.exp(lf0), lf0)
        return lf0
    else:
        print("Please specify either `numpy` or `tf`.")
        raise ValueError
