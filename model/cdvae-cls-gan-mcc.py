import numpy as np
import tensorflow as tf
from tensorflow.contrib import slim
from util.misc import ValueWindow
from util.layers import (GaussianKLD, GaussianLogDensity, GaussianSampleLayer,
                         Layernorm, conv2d_nchw_layernorm, lrelu,
                         kl_loss, log_loss, gradient_penalty_loss)

class CDVAECLSGAN(object):
    def __init__(self, arch, normalizers=None):
        '''
        Cross Domain Variational Auto Encoding
        Generative Adversarial Net with Adversarial classifier
        (CDVAE-CLS-GAN)
        Arguments:
            `arch`: network architecture (`dict`)
        '''
        self.arch = arch
        self.normalizers = normalizers

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

        self.mcc_dis = tf.make_template(
            'MCC_Discriminator',
            self.mcc_discriminator)

        self.latent_cls = tf.make_template(
            'Latent_Classifier',
            self.latent_classifier)

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

        x = tf.squeeze(x, axis=[-1]) # [N, C, n_frames]
        x = tf.transpose(x, perm=[0, 2, 1]) # [N, n_frames, C]
        return x

    def mcc_discriminator(self, x):
        net = self.arch['discriminator']['mcc']
        return self._discriminator(x, net)

    def _discriminator(self, x, net):
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
        x = tf.layers.dense(x, 1) # [N, n_frames, 1]
        return tf.reduce_mean(x, axis=1) #[N, 1]

    def latent_classifier(self, z):
        net = self.arch['classifier']['latent']
        
        x = tf.transpose(z, [0, 2, 1]) # [N, z_dim, n_frames]
        x = tf.expand_dims(x, -1) # [N, z_dim, n_frames, 1]
        
        for i, (o, k, s) in enumerate(zip(net['output'], net['kernel'], net['stride'])):
            with tf.variable_scope('Conv-{}'.format(i)):

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
        
        # carefully design the architecture so that now x has shape [N, C, n_frames, 1]
        batch_size, c, n_frames, w = x.get_shape().as_list()
        x = tf.transpose(x, perm=[0, 2, 1, 3]) # [N, n_frames, C, 1]
        x = tf.squeeze(x, axis=[-1]) # [N, n_frames, C]
        x = tf.layers.dense(x, self.arch['y_dim']) # [N, n_frames, 1]
        return tf.reduce_mean(x, axis=1) #[N, 1]

    def loss(self, data):

        x_sp = data['sp']
        x_mcc = data['mcc']
        y = data['speaker']
        label = tf.one_hot(tf.reduce_mean(y, axis=1, keep_dims=True), self.arch['y_dim'])
        
        # normalize input using mean/var
        x_sp_in_minmax = self.normalizers['sp']['minmax'].forward_process(x_sp)
        x_sp_in = tf.expand_dims(x_sp_in_minmax, 1) # insert channel dimension
        x_mcc_in_minmax = self.normalizers['mcc']['minmax'].forward_process(x_mcc)
        x_mcc_in = tf.expand_dims(x_mcc_in_minmax, 1) # insert channel dimension
       
        # forward pass
        # Use sp as source
        sp_z_mu, sp_z_lv = self.sp_enc(x_sp_in)
        z_sp = GaussianSampleLayer(sp_z_mu, sp_z_lv)
        x_sp_sp = self.sp_dec(z_sp, y)
        x_sp_mcc = self.mcc_dec(z_sp, y)

        cls_sp_logit = self.latent_cls(sp_z_mu)
        z_sp_pred = tf.nn.softmax(cls_sp_logit)
        sp_corr_pred = tf.equal(tf.argmax(z_sp_pred, 1), tf.reduce_mean(y, axis=1))

        # Use mcc as source
        mcc_z_mu, mcc_z_lv = self.mcc_enc(x_mcc_in)
        z_mcc = GaussianSampleLayer(mcc_z_mu, mcc_z_lv)
        x_mcc_sp = self.sp_dec(z_mcc, y)
        x_mcc_mcc = self.mcc_dec(z_mcc, y)
        
        x_mcc_mcc_NCHW = tf.expand_dims(x_mcc_mcc, axis=1)
        real_mcc_logit = self.mcc_dis(x_mcc_in)
        fake_mcc_logit = self.mcc_dis(x_mcc_mcc_NCHW)
        cls_mcc_logit = self.latent_cls(mcc_z_mu)
        z_mcc_pred = tf.nn.softmax(cls_mcc_logit)
        mcc_corr_pred = tf.equal(tf.argmax(z_mcc_pred, 1), tf.reduce_mean(y, axis=1))

        # loss
        kl_loss_sp = kl_loss(sp_z_mu, sp_z_lv)
        recon_loss_sp = log_loss(x_sp_in, x_sp_sp)
        cross_loss_sp2mcc = log_loss(x_mcc_in, x_sp_mcc)
        kl_loss_mcc = kl_loss(mcc_z_mu, mcc_z_lv)
        recon_loss_mcc = log_loss(x_mcc_in, x_mcc_mcc_NCHW)
        cross_loss_mcc2sp = log_loss(x_sp_in, x_mcc_sp)
        latent_loss = tf.reduce_mean(tf.abs(sp_z_mu - mcc_z_mu))

        gradient_penalty_mcc = gradient_penalty_loss(x_mcc_in, x_mcc_mcc, self.mcc_dis)
        cls_loss_sp = \
            tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
                labels=tf.stop_gradient(label), logits=cls_sp_logit))
        cls_loss_mcc = \
            tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
                labels=tf.stop_gradient(label), logits=cls_mcc_logit))
        
        acc = 0.5 * (tf.reduce_mean(tf.cast(sp_corr_pred, tf.float32)) + tf.reduce_mean(tf.cast(mcc_corr_pred, tf.float32)))

        loss = dict()
        loss['D_KL_sp'] = kl_loss_sp
        loss['D_KL_mcc'] = kl_loss_mcc
        loss['recon_sp'] = recon_loss_sp 
        loss['recon_mcc'] = recon_loss_mcc
        loss['cross_sp2mcc'] = cross_loss_sp2mcc 
        loss['cross_mcc2sp'] = cross_loss_mcc2sp
        loss['latent'] = latent_loss
        loss['wgan_mcc'] = tf.reduce_mean(fake_mcc_logit) - tf.reduce_mean(real_mcc_logit)
        loss['wgan_gp_mcc'] = gradient_penalty_mcc
        loss['cls_loss_sp'] = cls_loss_sp
        loss['cls_loss_mcc'] = cls_loss_mcc

        with tf.name_scope('Summary'):
            tf.summary.scalar('KL-div-sp', kl_loss_sp)
            tf.summary.scalar('KL-div-mcc', kl_loss_mcc)
            tf.summary.scalar('reconstruction-sp', recon_loss_sp)
            tf.summary.scalar('reconstruction-mcc', recon_loss_mcc)
            tf.summary.scalar('cross-sp2mcc', cross_loss_sp2mcc)
            tf.summary.scalar('cross-mcc2sp', cross_loss_mcc2sp)
            tf.summary.scalar('latent', latent_loss)
            tf.summary.scalar('wgan-mcc', loss['wgan_mcc'])
            tf.summary.scalar('wgan-gp-mcc', gradient_penalty_mcc)
            tf.summary.scalar('cls-sp', cls_loss_sp)
            tf.summary.scalar('cls-mcc', cls_loss_mcc)
            tf.summary.scalar('cls-accuracy', acc)
        
        return loss

    def get_train_log(self, step, time_window, loss_windows):
        msg = 'Iter {:05d}: '.format(step)
        msg += '{:.2} sec/step, '.format(time_window.average)
        msg += 'recon = {:.4} '.format(loss_windows['recon_sp'].average+loss_windows['recon_mcc'].average)
        msg += 'cross = {:.4} '.format(loss_windows['cross_sp2mcc'].average+loss_windows['cross_mcc2sp'].average)
        msg += 'KL = {:.4} '.format(loss_windows['D_KL_sp'].average+loss_windows['D_KL_mcc'].average)
        msg += 'latent = {:.5} '.format(loss_windows['latent'].average)
        msg += 'W = {:.3} '.format(loss_windows['wgan_mcc'].average+loss_windows['wgan_gp_mcc'].average)
        return msg
    

    def fetches(self, loss, opt): 
        """ define fetches
            update_fetches: get logging infos
            info_fetches: optimization only
        """
        vae_fetches_step_update = {
            'sp_decoder': opt['opt_sp_g_step_update'],
            'mcc_decoder': opt['opt_mcc_g'],
            'sp_encoder': opt['opt_sp_e'],
            'mcc_encoder': opt['opt_mcc_e'],
        }
        vae_fetches_no_step_update = {
            'sp_decoder': opt['opt_sp_g'],
            'mcc_decoder': opt['opt_mcc_g'],
            'sp_encoder': opt['opt_sp_e'],
            'mcc_encoder': opt['opt_mcc_e'],
        }
        cls_fetches = opt['opt_pre_c']
        gan_fetches = {
            'discriminator': opt['opt_d'],
            'classifier': opt['opt_c']
        }
        
        info_fetches = {
            'step': opt['global_step'],
        }
        for k in loss:
            info_fetches[k] = loss[k]

        return {
            'vae_step_update': vae_fetches_step_update,
            'vae_no_step_update': vae_fetches_no_step_update,
            'cls': cls_fetches,
            'gan': gan_fetches,
            'info': info_fetches,
        }

    def encode(self, x, feat_type):
        # sanity check
        if not feat_type in ['sp', 'mcc']:
            print('feature type does not match!')
            raise NotImplementedError

        # normalize input using mean/var
        x_in_minmax = self.normalizers[feat_type]['minmax'].forward_process(x) # [n_frames, dim]
        x_in = tf.expand_dims(tf.expand_dims(x_in_minmax, 0), 0) # [1, 1, n_frames, dim]

        if feat_type == 'sp':
            return self.sp_enc(x_in)
        elif feat_type == 'mcc':
            return self.mcc_enc(x_in)

    def decode(self, z, y, feat_type):
        # sanity check
        if not feat_type in ['sp', 'mcc']:
            print('feature type does not match!')
            raise NotImplementedError
        
        if feat_type == 'sp':
            xh = self.sp_dec(z, y)
        elif feat_type == 'mcc':
            xh = self.mcc_dec(z, y)
        
        xh = tf.squeeze(xh, 0)
        return self.normalizers[feat_type]['minmax'].backward_process(xh)
