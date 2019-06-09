import tensorflow as tf
from tensorflow.contrib import slim
from util.layers import (GaussianKLD, GaussianLogDensity, GaussianSampleLayer,
                         Layernorm, conv2d_nchw_layernorm, lrelu,
                         kl_loss, log_loss, gradient_penalty_loss)
import numpy as np

class CDVAEGAN(object):
    def __init__(self, arch):
        '''
        Cross Domain Variational Auto Encoding
        Generative Adversarial Net (CDVAE-GAN)
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

        self.sp_dis = tf.make_template(
            'SP_Discriminator',
            self.mcc_discriminator)

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

        # carefully design the architecture so that now x has shape [N, C, n_frames, 1]
        batch_size, c, n_frames, w = x.get_shape().as_list()
        x = tf.transpose(x, perm=[0, 2, 1, 3]) # [N, n_frames, C, 1]
        x = tf.squeeze(x, axis=[-1]) # [N, n_frames, C]
        z_mu = tf.layers.dense(x, self.arch['z_dim'], name='Dense-mu') # [N, n_frames, z_dim]
        z_lv = tf.layers.dense(x, self.arch['z_dim'], name='Dense-lv') # [N, n_frames, z_dim]
        return z_mu, z_lv

    def sp_decoder(self, z, y):
        net = self.arch['generator']['sp']
        return self._generator(z, y, net)
    
    def mcc_decoder(self, z, y):
        net = self.arch['generator']['mcc']
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

        return x  # [N, feat_dim, n_frames, 1]

    def sp_discriminator(self, x):
        net = self.arch['discriminator']['sp']
        return self._discriminator(x, net)

    def _discriminator(self, x, net):
        for i, (o, k, s) in enumerate(zip(net['output'], net['kernel'], net['stride'])):
            x = conv2d_nchw_layernorm(
                x, o, k, s, lrelu,
                name='Conv-{}'.format(i)
            )

        # carefully design the architecture so that now x has shape [N, C, n_frames, 1]
        batch_size, c, n_frames, w = x.get_shape().as_list()
        x = tf.transpose(x, perm=[0, 2, 1, 3]) # [N, n_frames, C, 1]
        x = tf.squeeze(x, axis=[-1]) # [N, n_frames, C]
        x = tf.layers.dense(x, 1) # [N, n_frames, 1]
        return tf.reduce_mean(x, axis=1) #[N, 1]

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
        
        real_sp_logit = self.sp_dis(x_sp)
        fake_sp_logit = self.sp_dis(x_sp_sp)

        gradient_penalty_sp = gradient_penalty_loss(x_sp, x_sp_sp, self.sp_dis) 
        

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
        loss['D_KL_sp'] = kl_loss_sp
        loss['D_KL_mcc'] = kl_loss_mcc
        loss['recon_sp'] = recon_loss_sp 
        loss['recon_mcc'] = recon_loss_mcc
        loss['cross_sp2mcc'] = cross_loss_sp2mcc 
        loss['cross_mcc2sp'] = cross_loss_mcc2sp
        loss['latent'] = latent_loss
        loss['wgan_sp'] = tf.reduce_mean(fake_sp_logit) - tf.reduce_mean(real_sp_logit)
        loss['wgan_gp_sp'] = gradient_penalty_sp

        with tf.name_scope('Summary'):
            tf.summary.scalar('KL-div-sp', kl_loss_sp)
            tf.summary.scalar('KL-div-mcc', kl_loss_mcc)
            tf.summary.scalar('reconstruction-sp', recon_loss_sp)
            tf.summary.scalar('reconstruction-mcc', recon_loss_mcc)
            tf.summary.scalar('cross-sp2mcc', cross_loss_sp2mcc)
            tf.summary.scalar('cross-mcc2sp', cross_loss_mcc2sp)
            tf.summary.scalar('latent', latent_loss)
            tf.summary.scalar('wgan-sp', loss['wgan_sp'])
            tf.summary.scalar('wgan-gp-sp', gradient_penalty_sp)
            
            tf.summary.histogram('x_sp_sp', x_sp_sp)
            tf.summary.histogram('x_mcc_mcc', x_mcc_mcc)
            tf.summary.histogram('x_sp', x_sp)
            tf.summary.histogram('x_mcc', x_mcc)
            
        return loss

    def validate(self, data):

        x_sp = data['sp']
        y = data['speaker']
        y = tf.expand_dims(y, 0)
        
        # Use sp as source
        z_sp, _ = self.sp_enc(x_sp)
        x_sp_sp = self.sp_dec(z_sp, y)

        recon_loss_sp = log_loss(x_sp, x_sp_sp)

        
        results = dict()
        results['recon_sp'] = recon_loss_sp 

        results['z_sp'] = z_sp
        results['x_sp_sp'] = x_sp_sp
        results['num_files'] = data['num_files']
        
        return results

    def get_train_log(self, result):
        msg = 'Iter {:05d}: '.format(result['step'])
        msg += 'recon = {:.4} '.format(result['recon_sp']+result['recon_mcc'])
        msg += 'cross = {:.4} '.format(result['cross_sp2mcc']+result['cross_mcc2sp'])
        msg += 'KL = {:.4} '.format(result['D_KL_sp']+result['D_KL_mcc'])
        msg += 'latent = {:.5} '.format(result['latent'])
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
            'sp_decoder': opt['opt_sp_g'],
            'mcc_decoder': opt['opt_mcc_g'],
            'sp_encoder': opt['opt_sp_e'],
            'mcc_encoder': opt['opt_mcc_e'],
        }
        gan_fetches = opt['opt_d']
        info_fetches = {
            'Dis': loss['wgan_sp'],
            "D_KL_sp": loss['D_KL_sp'],
            "D_KL_mcc": loss['D_KL_mcc'],
            "recon_sp": loss['recon_sp'],
            "recon_mcc": loss['recon_mcc'],
            "cross_sp2mcc": loss['cross_sp2mcc'],
            "cross_mcc2sp": loss['cross_mcc2sp'],
            "latent": loss['latent'],
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
    
    def mcc_encode(self, x):
        z_mu, _ = self.mcc_enc(x)
        return z_mu

    def mcc_decode(self, z, y):
        return self.mcc_dec(z, y)

    def encode(self, x):
        return self.sp_encode(x)

    def decode(self, z, y):
        return self.sp_decode(z, y)