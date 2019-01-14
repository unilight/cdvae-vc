import tensorflow as tf
from tensorflow.contrib import slim
from util.layers import (GaussianKLD, GaussianLogDensity, GaussianSampleLayer,
                         Layernorm, conv2d_nchw_layernorm, lrelu,
                         kl_loss, log_loss)
import numpy as np

class CDVAE(object):
    def __init__(self, arch):
        '''
        Cross Domain Variational Auto Encoder (CDVAE)
        Arguments:
            `arch`: network architecture (`dict`)
        '''
        self.arch = arch

        with tf.name_scope('SpeakerCode'):
            self.y_emb = self._l2_regularized_embedding(
                self.arch['y_dim'],
                self.arch['z_dim'],
                'y_embedding')

        self.sp_enc = tf.make_template(
            'SP_Encoder',
            self.sp_encoder)
        
        self.mcc_enc = tf.make_template(
            'MCC_Encoder',
            self.mcc_encoder)
        
        self.sp_dec = tf.make_template(
            'SP_Decoder',
            self.sp_decoder)
        
        self.mcc_dec = tf.make_template(
            'MCC_Decoder',
            self.mcc_decoder)

    def _merge(self, var_list, fan_out, l2_reg=1e-6):
        x = 0.
        with slim.arg_scope(
            [slim.fully_connected],
            num_outputs=fan_out,
            weights_regularizer=slim.l2_regularizer(l2_reg),
            normalizer_fn=None,
            activation_fn=None):
            for var in var_list:
                x = x + slim.fully_connected(var)
        return slim.bias_add(x)


    def _l2_regularized_embedding(self, n_class, h_dim, scope_name, var_name='y_emb'):
        with tf.variable_scope(scope_name):
            embeddings = tf.get_variable(
                name=var_name,
                shape=[n_class, h_dim],
                regularizer=slim.l2_regularizer(1e-6))
        return embeddings

    def sp_encoder(self, x):
        net = self.arch['encoder']['sp']
        return self._encoder(x, net)
    
    def mcc_encoder(self, x):
        net = self.arch['encoder']['mcc']
        return self._encoder(x, net)

    def _encoder(self, x, net):
        for i, (o, k, s) in enumerate(zip(net['output'], net['kernel'], net['stride'])):
            x = conv2d_nchw_layernorm(
                x, o, k, s, lrelu,
                name='Conv-{}'.format(i)
            )
        x = slim.flatten(x)
        z_mu = tf.layers.dense(x, self.arch['z_dim'], name='Dense-mu')
        z_lv = tf.layers.dense(x, self.arch['z_dim'], name='Dense-lv')
        return z_mu, z_lv

    def sp_decoder(self, z, y):
        net = self.arch['generator']['sp']
        return self._generator(z, y, net)
    
    def mcc_decoder(self, z, y):
        net = self.arch['generator']['mcc']
        return self._generator(z, y, net)

    def _generator(self, z, y, net):
        h, w, c = net['hwc']

        y = tf.nn.embedding_lookup(self.y_emb, y)
        with tf.variable_scope('Merge'):
            x = self._merge([z, y], h * w * c)

        x = tf.reshape(x, [-1, c, h, w])
        for i, (o, k, s) in enumerate(zip(net['output'], net['kernel'], net['stride'])):
            with tf.variable_scope('Conv-{}'.format(i)):
                x = tf.layers.conv2d_transpose(x, o, k, s,
                    padding='same',
                    data_format='channels_first',
                )
                if i < len(net['output']) -1:
                    x = Layernorm(x, [1, 2, 3], 'layernorm')
                    x = lrelu(x)
        return x

    def loss(self, data):

        x_sp = data['sp']
        x_mcc = data['mcc']
        y = data['speaker']
        
        # Use sp as source
        sp_z_mu, sp_z_lv = self.sp_enc(x_sp)
        z_sp = GaussianSampleLayer(sp_z_mu, sp_z_lv)
        x_sp_sp = self.sp_dec(z_sp, y)
        x_sp_mcc = self.mcc_dec(z_sp, y)

        kl_loss_sp = kl_loss(sp_z_mu, sp_z_lv)
        recon_loss_sp = log_loss(x_sp, x_sp_sp)
        cross_loss_sp2mcc = log_loss(x_mcc, x_sp_mcc)
       
        # Use mcc as source
        mcc_z_mu, mcc_z_lv = self.mcc_enc(x_mcc)
        z_mcc = GaussianSampleLayer(mcc_z_mu, mcc_z_lv)
        x_mcc_sp = self.sp_dec(z_mcc, y)
        x_mcc_mcc = self.mcc_dec(z_mcc, y)

        kl_loss_mcc = kl_loss(mcc_z_mu, mcc_z_lv)
        recon_loss_mcc = log_loss(x_mcc, x_mcc_mcc)
        cross_loss_mcc2sp = log_loss(x_sp, x_mcc_sp)
        
        # latent loss
        latent_loss = tf.reduce_mean(tf.abs(sp_z_mu - mcc_z_mu))

        loss = dict()
        loss['D_KL'] = (  kl_loss_sp
                        + kl_loss_mcc
                        )
        loss['recon'] = (  recon_loss_sp 
                         + recon_loss_mcc
                         )
        loss['cross'] = (  cross_loss_sp2mcc
                         + cross_loss_mcc2sp
                         )
        loss['latent'] = latent_loss

        loss['all'] = - loss['recon'] - loss['cross'] + loss['D_KL'] + loss['latent']

        tf.summary.scalar('KL-div-sp', kl_loss_sp)
        tf.summary.scalar('KL-div-mcc', kl_loss_mcc)
        tf.summary.scalar('reconstruction-sp', recon_loss_sp)
        tf.summary.scalar('reconstruction-mcc', recon_loss_mcc)
        tf.summary.scalar('cross-sp2mcc', cross_loss_sp2mcc)
        tf.summary.scalar('cross-mcc2sp', cross_loss_mcc2sp)
        tf.summary.scalar('latent', latent_loss)

        tf.summary.histogram('x_sp_sp', x_sp_sp)
        tf.summary.histogram('x_mcc_mcc', x_mcc_mcc)
        tf.summary.histogram('x_sp', x_sp)
        tf.summary.histogram('x_mcc', x_mcc)
        return loss

    def validate(self, data):

        x_sp = data['sp']
        x_mcc = data['mcc']
        y = data['speaker']
        
        # Use sp as source
        z_sp, _ = self.sp_enc(x_sp)
        x_sp_sp = self.sp_dec(z_sp, y)
        x_sp_mcc = self.mcc_dec(z_sp, y)

        recon_loss_sp = log_loss(x_sp, x_sp_sp)
       
        # Use mcc as source
        z_mcc, _ = self.mcc_enc(x_mcc)
        x_mcc_sp = self.sp_dec(z_mcc, y)
        x_mcc_mcc = self.mcc_dec(z_mcc, y)

        recon_loss_mcc = log_loss(x_mcc, x_mcc_mcc)
        
        results = dict()
        results['recon_sp'] = recon_loss_sp 
        results['recon_mcc'] = recon_loss_mcc

        results['z_sp'] = z_sp
        results['z_mcc'] = z_mcc
        results['x_mcc_mcc'] = x_mcc_mcc
        results['x_mcc_sp'] = x_mcc_sp
        results['x_sp_mcc'] = x_sp_mcc
        results['x_sp_sp'] = x_sp_sp
        results['num_files'] = data['num_files']
        
        return results

    def get_train_log(self, result):
        msg = 'Iter {:05d}: '.format(result['step'])
        msg += 'recon = {:.4} '.format(result['recon'])
        msg += 'cross = {:.4} '.format(result['cross'])
        msg += 'KL = {:.4} '.format(result['D_KL'])
        msg += 'latent = {:.5} '.format(result['latent'])
        return msg
    
    def get_valid_log(self, step, result_all):
        valid_loss_all = [loss['recon'] for loss in result_all]
        valid_loss_avg = np.mean(np.array(valid_loss_all))
        msg = 'Validation in Iter {:05d}: '.format(step)
        msg += 'recon = {:.4} '.format(valid_loss_avg)
        return msg

    def fetches(self, loss, valid, opt): 
        """ define fetches
            update_fetches: get logging infos
            info_fetches: optimization only
            valid_fetches: for validation
        """
        update_fetches = opt['opt']
        info_fetches = {
            "D_KL": loss['D_KL'],
            "recon": loss['recon'],
            "cross": loss['cross'],
            "latent": loss['latent'],
            "opt": opt['opt'],
            "step": opt['global_step'],
        }
        valid_fetches = {
            "recon": valid['recon_mcc'],
            "step": opt['global_step'],
        }

        return {
            'update': update_fetches,
            'info': info_fetches,
            'valid': valid_fetches,
        }

    def sp_encode(self, x):
        z_mu, _ = self.sp_enc(x)
        return z_mu

    def sp_decode(self, z, y):
        return self.sp_dec(z, y)
    
    def mcc_encode(self, x):
        z_mu, _ = self.mcc_enc(x)
        return z_mu

    def mcc_decode(self, z, y):
        return self.mcc_dec(z, y)

    def encode(self, x):
        return self.mcc_encode(x)

    def decode(self, z, y):
        return self.mcc_decode(z, y)
