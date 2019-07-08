import numpy as np
import tensorflow as tf
from tensorflow.contrib import slim
from util.misc import ValueWindow
from util.layers import (GaussianKLD, GaussianLogDensity, GaussianSampleLayer,
                         Layernorm, conv2d_nchw_layernorm, lrelu,
                         kl_loss, log_loss)

class VAE(object):
    def __init__(self, arch, normalizers=None):
        '''
        Variational Auto Encoder (VAE)
        Arguments:
            `arch`: network architecture (`dict`)
        '''
        self.arch = arch
        self.normalizers = normalizers
        self.feat_type = arch['feat_type']

        with tf.name_scope('SpeakerCode'):
            self.y_emb = self._l2_regularized_embedding(
                self.arch['y_dim'],
                self.arch['z_dim'],
                'y_embedding')

        self.enc = tf.make_template(
            'Encoder',
            self.encoder)
        
        self.dec = tf.make_template(
            'Decoder',
            self.decoder)

    def _l2_regularized_embedding(self, n_class, h_dim, scope_name, var_name='y_emb'):
        with tf.variable_scope(scope_name):
            embeddings = tf.get_variable(
                name=var_name,
                shape=[n_class, h_dim],
                regularizer=slim.l2_regularizer(1e-6))
        return embeddings

    def encoder(self, x):
        net = self.arch['encoder'][self.feat_type]
        return self._encoder(x, net)

    def _encoder(self, x, net):
        x = tf.transpose(x, perm=[0, 3, 2, 1]) # [N, d, n_frames, 1]
        
        for i, (o, k, s) in enumerate(zip(net['output'], net['kernel'], net['stride'])):
            x = conv2d_nchw_layernorm(
                x, o, k, s, lrelu,
                name='Conv-{}'.format(i)
            )

        # carefully design the architecture so that now x has shape [N, C, n_frames, 1]
        batch_size, c, n_frames, w = x.get_shape().as_list()
        x = tf.transpose(x, perm=[0, 2, 1, 3]) # [N, n_frames, C, 1]
        x = tf.squeeze(x, axis=[-1]) # [N, n_frames, C]
        z_mu = tf.layers.dense(x, self.arch['z_dim'], name='Dense-mu') # [N, n_frames, z_dim]
        z_lv = tf.layers.dense(x, self.arch['z_dim'], name='Dense-lv') # [N, n_frames, z_dim]
        return z_mu, z_lv

    def decoder(self, z, y):
        net = self.arch['generator'][self.feat_type]
        return self._generator(z, y, net)

    def _generator(self, z, y, net):

        x = tf.expand_dims(z, 1) # [N, 1, n_frames, z_dim]
        x = tf.transpose(x, perm=[0, 3, 2, 1]) # [N, z_dim, n_frames, 1]
        y = tf.nn.embedding_lookup(self.y_emb, y) # [N, n_frames, y_emb_dim]
        y = tf.expand_dims(y, 1) # [N, 1, n_frames, y_emb_dim]

        for i, (o, k, s) in enumerate(zip(net['output'], net['kernel'], net['stride'])):
            with tf.variable_scope('Conv-{}'.format(i)):

                # concat y along channel axis
                y_transpose = tf.transpose(y, perm=[0, 3, 2, 1]) # [N, y_emb_dim, n_frames, 1]
                x = tf.concat([x, y_transpose], axis=1) # [N, channels + y_emb_dim, n_frames, 1]

                if i < len(net['output']) -1:
                    x_tanh = tf.layers.conv2d_transpose(x, o, k, s,
                        padding='same',
                        data_format='channels_first',
                    )
                    x_tanh = Layernorm(x_tanh, [1, 2, 3], 'Layernorm-{}-tanh'.format(i))
                    
                    x_sigmoid = tf.layers.conv2d_transpose(x, o, k, s,
                        padding='same',
                        data_format='channels_first',
                    )
                    x_sigmoid = Layernorm(x_sigmoid, [1, 2, 3], 'Layernorm-{}-sigmoid'.format(i))

                    # GLU
                    with tf.variable_scope('GLU'):
                        x = tf.sigmoid(x_sigmoid) * tf.tanh(x_tanh)
                else:
                    x = tf.layers.conv2d_transpose(x, o, k, s,
                        padding='same',
                        data_format='channels_first',
                    )

        x = tf.squeeze(x, axis=[-1]) # [N, C, n_frames]
        x = tf.transpose(x, perm=[0, 2, 1]) # [N, n_frames, C]
        return x

    def loss(self, data):

        x = data[self.feat_type]
        y = data['speaker']
        
        # normalize input using mean/var
        x_in_minmax = self.normalizers[self.feat_type]['minmax'].forward_process(x)
        x_in = tf.expand_dims(x_in_minmax, 1) # insert channel dimension
       
        # forward pass
        z_mu, z_lv = self.enc(x_in)
        z = GaussianSampleLayer(z_mu, z_lv)
        xb = self.dec(z, y)

        # loss
        KL_loss = kl_loss(z_mu, z_lv)
        recon_loss = log_loss(x_in_minmax, xb)
        
        loss = dict()
        loss['D_KL'] = KL_loss
        loss['recon'] = recon_loss

        loss['all'] = - loss['recon'] + loss['D_KL']

        # summary
        tf.summary.scalar('KL-div-sp', KL_loss)
        tf.summary.scalar('reconstruction-sp', recon_loss)
        return loss


    def get_train_log(self, step, time_window, loss_windows):
        msg = 'Iter {:05d}: '.format(step)
        msg += '{:.2} sec/step, '.format(time_window.average)
        msg += 'recon = {:.4} '.format(loss_windows['recon'].average)
        msg += 'KL = {:.4} '.format(loss_windows['D_KL'].average)
        return msg

    def fetches(self, loss, opt): 
        """ define fetches
            update_fetches: get logging infos
            info_fetches: optimization only
        """
        update_fetches = opt['opt']
        info_fetches = {
            "D_KL": loss['D_KL'],
            "recon": loss['recon'],
            "all": loss['all'],
            "opt": opt['opt'],
            "step": opt['global_step'],
        }

        return {
            'update': update_fetches,
            'info': info_fetches,
        }

    def encode(self, x, feat_type):
        # sanity check
        if not feat_type == self.feat_type:
            print('feature type does not match!')
            raise NotImplementedError

        # normalize input using mean/var
        x_in_minmax = self.normalizers[self.feat_type]['minmax'].forward_process(x) # [n_frames, dim]
        x_in = tf.expand_dims(tf.expand_dims(x_in_minmax, 0), 0) # [1, 1, n_frames, dim]

        return self.enc(x_in)

    def decode(self, z, y, feat_type):
        # sanity check
        if not feat_type == self.feat_type:
            print('feature type does not match!')
            raise NotImplementedError
        
        xh = self.dec(z, y)
        xh = tf.squeeze(xh, 0)
        return self.normalizers[self.feat_type]['minmax'].backward_process(xh)
