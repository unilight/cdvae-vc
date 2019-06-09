import logging
import os

import numpy as np
import tensorflow as tf
from trainer.base import Trainer
from util.wrapper import load

class CDVAEGANTrainer(Trainer):
    '''
    Trainer for training CDVAEGAN 
    '''

    def _optimize(self):
        """ get the following operators:
            opt: update operator
            global_step: global step
        """
        
        global_step = tf.train.get_or_create_global_step()
        lr = self.arch['training']['lr']
        b1 = self.arch['training']['beta1']
        b2 = self.arch['training']['beta2']
        optimizer_d = tf.train.AdamOptimizer(lr, b1, b2)
        optimizer_g = tf.train.AdamOptimizer(lr, b1, b2)

        var_list = tf.trainable_variables()
        sp_e_vars = [v for v in var_list if 'SP_Encoder' in v.name]
        mcc_e_vars = [v for v in var_list if 'MCC_Encoder' in v.name]
        sp_g_vars = [v for v in var_list if 'SP_Decoder' in v.name]
        mcc_g_vars = [v for v in var_list if 'MCC_Decoder' in v.name]
        sp_d_vars = [v for v in var_list if 'SP_Discriminator' in v.name]
        mcc_d_vars = [v for v in var_list if 'MCC_Discriminator' in v.name]

        r_sp = tf.placeholder(shape=[], dtype=tf.float32)
        r_mcc = tf.placeholder(shape=[], dtype=tf.float32)
        gp_weight = self.arch['training']['gp_weight']

        obj_sp_Ez = self.loss['D_KL_sp'] - self.loss['recon_sp'] - self.loss['cross_sp2mcc'] + self.loss['latent']
        obj_mcc_Ez = self.loss['D_KL_mcc'] - self.loss['recon_mcc'] - self.loss['cross_mcc2sp'] + self.loss['latent']
        obj_sp_Gx = -r_sp * self.loss['wgan_sp'] - self.loss['recon_sp'] - self.loss['cross_mcc2sp']
        obj_mcc_Gx = -r_mcc * self.loss['wgan_mcc'] - self.loss['recon_mcc'] - self.loss['cross_sp2mcc']
        obj_sp_D = self.loss['wgan_sp'] + gp_weight * self.loss['wgan_gp_sp']
        obj_mcc_D = self.loss['wgan_mcc'] + gp_weight * self.loss['wgan_gp_mcc']

        opt_sp_d = optimizer_d.minimize(obj_sp_D, var_list=sp_d_vars)
        opt_mcc_d = optimizer_d.minimize(obj_mcc_D, var_list=mcc_d_vars)
        opt_ds = [opt_sp_d, opt_mcc_d]

        opt_sp_g = optimizer_g.minimize(obj_sp_Gx, var_list=sp_g_vars, global_step=global_step)
        opt_mcc_g = optimizer_g.minimize(obj_mcc_Gx, var_list=mcc_g_vars)
        opt_sp_e = optimizer_g.minimize(obj_sp_Ez, var_list=sp_e_vars)
        opt_mcc_e = optimizer_g.minimize(obj_mcc_Ez, var_list=mcc_e_vars)

        return {
            'opt_d': opt_ds,
            'opt_sp_g': opt_sp_g,
            'opt_mcc_g': opt_mcc_g,
            'opt_sp_e': opt_sp_e,
            'opt_mcc_e': opt_mcc_e,
            'gamma_sp': r_sp,
            'gamma_mcc': r_mcc,
            'global_step': global_step
        }

    def train(self):

        # get fetches
        fetches = self.model.fetches(self.loss, self.valid, self.opt)

        # get session (defined in initialization)
        sess = self.sess
            
        # restore 
        self.restore()

        # Iterate through training steps
        while not sess.should_stop():

            # update global step
            step = tf.train.global_step(sess, self.opt['global_step'])
                
            if (step+1) <= self.arch['training']['vae_iter']:
                feed_dict = {
                    self.opt['gamma_sp']: 0.,
                    self.opt['gamma_mcc']: 0.}

                # Display progress when reached a certain frequency
                if (step+1) % self.arch['training']['log_freq'] == 0:
                    _, results = sess.run([fetches['vae'], fetches['info']], feed_dict=feed_dict)
                    msg = self.model.get_train_log(results)
                    self.print_log(msg)

                    """
                    # validation
                    valid_loss_all = []
                    for _ in range(self.valid['num_files']):
                        results = sess.run(fetches['valid'])
                        valid_loss_all.append(results)
                    
                    msg = self.model.get_valid_log(results['step'], valid_loss_all)
                    self.print_log(msg)
                    """
                else:
                    _ = sess.run(fetches['vae'], feed_dict=feed_dict)
                    
            else:
                feed_dict = {
                    self.opt['gamma_sp']: self.arch['training']['gamma_sp'],
                    self.opt['gamma_mcc']: self.arch['training']['gamma_mcc']}

                nIterD = self.arch['training']['nIterD']

                for _ in range(nIterD):
                    sess.run(fetches['gan'])

                # Display progress when reached a certain frequency
                if (step+1) % self.arch['training']['log_freq'] == 0:
                    _, results = sess.run([fetches['vae'], fetches['info']], feed_dict=feed_dict)
                    msg = self.model.get_train_log(results)
                    self.print_log(msg)
                    
                    """
                    # validation
                    valid_loss_all = []
                    for _ in range(self.valid['num_files']):
                        results = sess.run(fetches['valid'])
                        valid_loss_all.append(results)
                    
                    msg = self.model.get_valid_log(results['step'], valid_loss_all)
                    self.print_log(msg)
                    """

                else:
                    _ = sess.run(fetches['vae'], feed_dict=feed_dict)