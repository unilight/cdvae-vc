# import numpy as np
from math import pi
import tensorflow as tf
from tensorflow.contrib import slim

EPSILON = tf.constant(1e-6, dtype=tf.float32)
PI = tf.constant(pi, dtype=tf.float32)

def Layernorm(x, axis, name):
    '''
    Layer normalization (Ba, 2016)
    J: Z-normalization using all nodes of the layer on a per-sample basis.

    Input:
        `x`: channel_first/NCHW format! (or fully-connected)
        `axis`: list
        `name`: must be assigned
    
    Example:
        ```python
        axis = [1, 2, 3]
        x = tf.random_normal([64, 3, 10, 10])
        name = 'D_layernorm'
        ```
    Return:
        (x - u)/s * scale + offset

    Source: 
        https://github.com/igul222/improved_wgan_training/blob/master/tflib/ops/layernorm.py
    '''
    mean, var = tf.nn.moments(x, axis, keep_dims=True)
    n_neurons = x.get_shape().as_list()[axis[0]]
    offset = tf.get_variable(
        name+'.offset',
        shape=[n_neurons] + [1 for _ in range(len(axis) -1)],
        initializer=tf.zeros_initializer
    )
    scale = tf.get_variable(
        name+'.scale',
        shape=[n_neurons] + [1 for _ in range(len(axis) -1)],
        initializer=tf.ones_initializer
    )
    return tf.nn.batch_normalization(x, mean, var, offset, scale, 1e-5)


def conv2d_nchw_layernorm(x, o, k, s, activation, name):
    '''
    Input:
        `x`: input in NCHW format
        `o`: num of output nodes
        `k`: kernel size
        `s`: stride
    '''
    with tf.variable_scope(name):
        x = tf.layers.conv2d(
            inputs=x,
            filters=o,
            kernel_size=k,
            strides=s,
            padding='same',
            data_format='channels_first',
            name=name,
        )
        x = Layernorm(x, [1, 2, 3], 'layernorm')
        return activation(x)

def lrelu(x, leak=0.02, name="lrelu"):
    ''' Leaky ReLU '''
    return tf.maximum(x, leak*x, name=name)


def GaussianSampleLayer(z_mu, z_lv, name='GaussianSampleLayer'):
    with tf.name_scope(name):
        eps = tf.random_normal(tf.shape(z_mu))
        std = tf.sqrt(tf.exp(z_lv))
        return tf.add(z_mu, tf.multiply(eps, std))


def GaussianLogDensity(x, mu, log_var, name='GaussianLogDensity'):
    with tf.name_scope(name):
        c = tf.log(2. * PI)
        var = tf.exp(log_var)
        x_mu2 = tf.square(x - mu)   # [Issue] not sure the dim works or not?
        x_mu2_over_var = tf.div(x_mu2, var + EPSILON)
        log_prob = -0.5 * (c + log_var + x_mu2_over_var)
        log_prob = tf.reduce_sum(log_prob, -1)   # keep_dims=True,
        return log_prob


def GaussianKLD(mu1, lv1, mu2, lv2):
    ''' Kullback-Leibler divergence of two Gaussians
        *Assuming that each dimension is independent
        mu: mean
        lv: log variance
        Equation: http://stats.stackexchange.com/questions/7440/kl-divergence-between-two-univariate-gaussians
    '''
    with tf.name_scope('GaussianKLD'):
        v1 = tf.exp(lv1)
        v2 = tf.exp(lv2)
        mu_diff_sq = tf.square(mu1 - mu2)
        dimwise_kld = .5 * (
            (lv2 - lv1) + tf.div(v1 + mu_diff_sq, v2 + EPSILON) - 1.)
        return tf.reduce_sum(dimwise_kld, -1)

def kl_loss(z_mu, z_lv):
    return tf.reduce_mean(
            GaussianKLD(
                slim.flatten(z_mu),
                slim.flatten(z_lv),
                slim.flatten(tf.zeros_like(z_mu)),
                slim.flatten(tf.zeros_like(z_lv)),
            )
        )

def log_loss(x, xh):
    return tf.reduce_mean(
            GaussianLogDensity(
                slim.flatten(x),
                slim.flatten(xh),
                tf.zeros_like(slim.flatten(xh))),
        )

def gradient_penalty_loss(x, xh, discriminator):
    batch_size, _, _, _ = x.get_shape().as_list()
        
    # expand dims because the output of the network  is squeezed. TODO: fix this by not squeezing decoder outputs?
    xh_NCHW = tf.expand_dims(xh, axis=1)

    alpha_dist = tf.contrib.distributions.Uniform(low=0., high=1.)
    alpha = alpha_dist.sample((batch_size, 1, 1, 1))
    interpolated = x + alpha * (xh_NCHW - x)
    inte_logit = discriminator(interpolated)
    gradients = tf.gradients(inte_logit, [interpolated,])[0]
    grad_l2 = tf.sqrt(tf.reduce_sum(tf.square(gradients), axis=[1,2,3]))
    gradient_penalty = tf.reduce_mean((grad_l2-1)**2)
    return gradient_penalty
