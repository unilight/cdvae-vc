import tensorflow as tf
from tensorflow.contrib import slim
from util.layers import (GaussianKLD, GaussianLogDensity, GaussianSampleLayer,
                         Layernorm, conv2d_nchw_layernorm, lrelu,
                         kl_loss, log_loss, gradient_penalty_loss)
import numpy as np

class VAESGAN(object):
    def __init__(self, arch, is_training = False):
        '''
        Variational Auto Encoding 
        Supervised Generative Adversarial Net (VAE-SGAN)

        Arguments:
            `arch`: network architecture (`dict`)
        '''
        self.arch = arch
        self.is_training = is_training

        with tf.name_scope('SpeakerCode'):
            self.y_emb = self._l2_regularized_embedding(
                self.arch['y_dim'],
                self.arch['z_dim'],
                'y_embedding')

        self.sp_enc = tf.make_template(
            'SP_Encoder',
            self.sp_encoder)
        
        self.sp_dec = tf.make_template(
            'SP_Decoder',
            self.sp_decoder)

        self.sp_dis = tf.make_template(
            'SP_Discriminator',
            self.sp_discriminator)

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

    def _encoder(self, x, net):

        with slim.arg_scope(
            [slim.batch_norm],
            scale=True, scope='BN',
            updates_collections=None,
            decay=0.9, epsilon=1e-5,  # [TODO] Test these hyper-parameters
            is_training=self.is_training):
            with slim.arg_scope(
                [slim.conv2d],
                weights_regularizer=slim.l2_regularizer(net['l2-reg']),
                normalizer_fn=slim.batch_norm,
                activation_fn=lrelu):

                for i, (o, k, s) in enumerate(zip(net['output'], net['kernel'], net['stride'])):
                    x = slim.conv2d(
                        x, o, k, s, 
                        data_format='NCHW'
                    )

        x = slim.flatten(x)

        with slim.arg_scope(
            [slim.fully_connected],
            num_outputs=self.arch['z_dim'],
            weights_regularizer=slim.l2_regularizer(net['l2-reg']),
            normalizer_fn=None,
            activation_fn=None):
            z_mu = slim.fully_connected(x)
            z_lv = slim.fully_connected(x)
        return z_mu, z_lv

    def sp_decoder(self, z, y):
        net = self.arch['generator']['sp']
        return self._generator(z, y, net)
    
    def _generator(self, z, y, net):
        h, w, c = net['hwc']

        y = tf.nn.embedding_lookup(self.y_emb, y)
        with tf.variable_scope('Merge'):
            x = self._merge([z, y], net['merge_dim'])

        x = lrelu(x)
        with slim.arg_scope(
            [slim.batch_norm],
            scale=True, scope='BN',
            updates_collections=None,
            decay=0.9, epsilon=1e-5,
            is_training=self.is_training):

            x = slim.fully_connected(
                x,
                h * w * c,
                normalizer_fn=slim.batch_norm,
                activation_fn=lrelu)

            x = tf.reshape(x, [-1, c, h, w])

            with slim.arg_scope(
                [slim.conv2d_transpose],
                weights_regularizer=slim.l2_regularizer(net['l2-reg']),
                normalizer_fn=slim.batch_norm,
                activation_fn=lrelu):

                for i, (o, k, s) in enumerate(zip(net['output'], net['kernel'], net['stride'])):
                    if i < (len(net['output'])-1):
                        x = slim.conv2d_transpose(
                            x, o, k, s,
                            data_format='NCHW'
                            )
                    else:
                        # Don't apply BN for the last layer of G
                        x = slim.conv2d_transpose(
                            x, o, k, s,
                            normalizer_fn=None,
                            activation_fn=None,
                            data_format='NCHW'
                            )
                        x = tf.nn.tanh(x)

        return x
    
    def sp_discriminator(self, x):
        net = self.arch['discriminator']['sp']
        return self._discriminator(x, net)
    
    def _discriminator(self, x, net):
        intermediate = list()
        intermediate.append(x)

        with slim.arg_scope(
            [slim.batch_norm],
            scale=True, scope='BN',
            updates_collections=None,
            decay=0.9, epsilon=1e-5,
            is_training=self.is_training):
            with slim.arg_scope(
                [slim.conv2d],
                weights_regularizer=slim.l2_regularizer(net['l2-reg']),
                normalizer_fn=slim.batch_norm,
                activation_fn=lrelu):
                
                for i, (o, k, s) in enumerate(zip(net['output'], net['kernel'], net['stride'])):
                    if i == 0:
                        # Radford: [do] not applying batchnorm to the discriminator input layer
                        x = slim.conv2d(
                            x, o, k, s, 
                            normalizer_fn=None,
                            data_format='NCHW'
                            )
                        intermediate.append(x)
                    else:
                        x = slim.conv2d(
                            x, o, k, s, 
                            data_format='NCHW'
                            )
                        intermediate.append(x)

        # Don't apply BN for the last layer
        x = slim.flatten(x)
        x = slim.fully_connected(
            x,
            1,
            weights_regularizer=slim.l2_regularizer(net['l2-reg']),
            activation_fn=None)

        return x

    ###################################################################

    def loss(self, data):
        """
        For VAE-SGAN, `data` is a list: [src, trg]
        """
        data_s, data_t = data

        xs_sp = data_s['sp']
        ys = data_s['speaker']
        xt_sp = data_t['sp']
        yt = data_t['speaker']
        
        # encoding source
        sp_zs_mu, sp_zs_lv = self.sp_enc(xs_sp)
        zs_sp = GaussianSampleLayer(sp_zs_mu, sp_zs_lv)
        
        # encoding target
        sp_zt_mu, sp_zt_lv = self.sp_enc(xt_sp)
        zt_sp = GaussianSampleLayer(sp_zt_mu, sp_zt_lv)

        # decoding
        xs_sp_sp = self.sp_dec(zs_sp, ys)  # source recon
        xt_sp_sp = self.sp_dec(zt_sp, yt)  # target recon
        xc_sp_sp = self.sp_dec(zs_sp, yt)  # conversion

        # discriminating
        real_logit = self.sp_dis(xt_sp)
        fake_logit = self.sp_dis(xc_sp_sp)

        # KL loss
        s_kl_loss_sp = kl_loss(sp_zs_mu, sp_zs_lv)
        t_kl_loss_sp = kl_loss(sp_zt_mu, sp_zt_lv)

        # reconstruction loss
        s_recon_loss_sp = log_loss(xs_sp, xs_sp_sp)
        t_recon_loss_sp = log_loss(xt_sp, xt_sp_sp)
        
        gradient_penalty = gradient_penalty_loss(xt_sp, xc_sp_sp, self.sp_dis)

        loss = dict()
        loss['D_KL'] = (  s_kl_loss_sp
                        + t_kl_loss_sp
                        ) / 2.0
        loss['recon'] = (  s_recon_loss_sp 
                         + t_recon_loss_sp
                         ) / 2.0
        loss['wgan'] = tf.reduce_mean(fake_logit) - tf.reduce_mean(real_logit)
        loss['wgan_gp'] = gradient_penalty

        tf.summary.scalar('KL-div-sp', loss['D_KL'])
        tf.summary.scalar('reconstruction-sp', loss['recon'])

        return loss

    def validate(self, data):
        """
        For VAE-SGAN, `data` is a list: [src]
        We only examine self reconstruction here.
        """

        x_sp = data[0]['sp']
        y = data[0]['speaker']
        
        # Use sp as source
        z_sp, _ = self.sp_enc(x_sp)
        x_sp_sp = self.sp_dec(z_sp, y)

        recon_loss_sp = log_loss(x_sp, x_sp_sp)
        
        results = dict()
        results['recon_sp'] = recon_loss_sp 

        results['z_sp'] = z_sp
        results['x_sp_sp'] = x_sp_sp
        results['num_files'] = data[0]['num_files']
        
        return results
        
    def get_train_log(self, result):
        msg = 'Iter {:05d}: '.format(result['step'])
        msg += 'recon = {:.4} '.format(result['recon'])
        msg += 'KL = {:.4} '.format(result['D_KL'])
        msg += 'W = {:.3} '.format(result['Dis'])
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
        vae_fetches = {
                'decoder': opt['opt_g'],
                'encoder': opt['opt_e'],
        }
        gan_fetches = opt['opt_d']
        info_fetches = {
            'Dis': loss['wgan'],
            "D_KL": loss['D_KL'],
            "recon": loss['recon'],
            'step': opt['global_step'],
        }
        valid_fetches = {
            "recon": valid['recon_sp'],
            "step": opt['global_step'],
        }

        return {
            'vae': vae_fetches,
            'gan': gan_fetches,
            'info': info_fetches,
            'valid': valid_fetches,
        }

    def sp_encode(self, x):
        z_mu, _ = self.sp_enc(x)
        return z_mu

    def sp_decode(self, z, y):
        return self.sp_dec(z, y)

    def sp_discriminate(self, x):
        return self.sp_dis(x)

    def encode(self, x):
        return self.sp_encode(x)

    def decode(self, z, y):
        return self.sp_decode(z, y)
