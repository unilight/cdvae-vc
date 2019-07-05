import logging
import os

import numpy as np
import tensorflow as tf
from trainer.base import Trainer
from util.wrapper import ValueWindow
import time

class CDVAECLSGANTrainer(Trainer):
    '''
    Trainer for training CDVAE-CLS-GAN 
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
        d_vars = [v for v in var_list if 'MCC_Discriminator' in v.name]
        c_vars = [v for v in var_list if 'Latent_Classifier' in v.name]
        spk_vars = [v for v in var_list if 'SpeakerCode' in v.name]
        sp_g_vars = sp_g_vars + spk_vars
        mcc_g_vars = mcc_g_vars + spk_vars

        r = tf.placeholder(shape=[], dtype=tf.float32)
        l = tf.placeholder(shape=[], dtype=tf.float32)
        gp_weight = self.arch['training']['gp_weight']

        obj_sp_Ez = self.loss['D_KL_sp'] - self.loss['recon_sp'] - self.loss['cross_sp2mcc'] + self.loss['latent'] - l * self.loss['cls_loss_sp']
        obj_mcc_Ez = self.loss['D_KL_mcc'] - self.loss['recon_mcc'] - self.loss['cross_mcc2sp'] + self.loss['latent'] - l * self.loss['cls_loss_mcc']
        obj_sp_Gx = - self.loss['recon_sp'] - self.loss['cross_mcc2sp']
        obj_mcc_Gx = -r * self.loss['wgan_mcc'] - self.loss['recon_mcc'] - self.loss['cross_sp2mcc']
        obj_D = self.loss['wgan_mcc'] + gp_weight * self.loss['wgan_gp_mcc']
        obj_C = self.loss['cls_loss_sp'] + self.loss['cls_loss_mcc']

        opt_d = optimizer_d.minimize(obj_D, var_list=d_vars)

        opt_c = optimizer_d.minimize(obj_C, var_list=c_vars)
        opt_pre_c = optimizer_d.minimize(obj_C, var_list=c_vars, global_step=global_step)

        opt_sp_g_step_update = optimizer_g.minimize(obj_sp_Gx, var_list=sp_g_vars, global_step=global_step)
        opt_sp_g = optimizer_g.minimize(obj_sp_Gx, var_list=sp_g_vars)
        opt_mcc_g = optimizer_g.minimize(obj_mcc_Gx, var_list=mcc_g_vars)
        opt_sp_e = optimizer_g.minimize(obj_sp_Ez, var_list=sp_e_vars)
        opt_mcc_e = optimizer_g.minimize(obj_mcc_Ez, var_list=mcc_e_vars)

        return {
            'opt_d': opt_d,
            'opt_sp_g_step_update': opt_sp_g_step_update,
            'opt_sp_g': opt_sp_g,
            'opt_mcc_g': opt_mcc_g,
            'opt_sp_e': opt_sp_e,
            'opt_mcc_e': opt_mcc_e,
            'opt_c': opt_c,
            'opt_pre_c': opt_pre_c,
            'gamma': r,
            'lambda': l,
            'global_step': global_step
        }

    def train(self):

        # get fetches
        fetches = self.model.fetches(self.loss, self.opt)

        # init windows
        time_window = ValueWindow(100)
        loss_window = ValueWindow(100)

        # get session (defined in initialization)
        sess = self.sess
            
        # restore 
        self.restore()

        start_time = time.time()
        # Iterate through training steps
        while not sess.should_stop():

            # update global step
            step = tf.train.global_step(sess, self.opt['global_step'])
            
            ##########################
            # Phase 1: train the VAE #
            ##########################
            if (step+1) <= self.arch['training']['vae_iter']:
                feed_dict = {
                    self.opt['gamma']: 0.,
                    self.opt['lambda']: 0.
                    }

                # Display progress when reached a certain frequency
                if (step+1) % self.arch['training']['log_freq'] == 0:
                    _, results = sess.run([fetches['vae_step_update'], fetches['info']], feed_dict=feed_dict)
                
                    # update windows
                    elapsed_time = (time.time()-start_time) / step
                    self.update_windows(elapsed_time, results)
               
                    # log
                    msg = self.model.get_train_log(results['step'], self.time_window, self.loss_windows)
                    self.print_log(msg)

                else:
                    _ = sess.run(fetches['vae_step_update'], feed_dict=feed_dict)

            ##########################
            # Phase 2: train the CLS #
            ##########################
            elif (step+1) > self.arch['training']['vae_iter'] and (step+1) <= (self.arch['training']['vae_iter'] + self.arch['training']['cls_iter']):
                feed_dict = {
                    self.opt['gamma']: 0.,
                    self.opt['lambda']: 0.
                    }

                # Display progress when reached a certain frequency
                if (step+1) % self.arch['training']['log_freq'] == 0:
                    _, results = sess.run([fetches['cls'], fetches['info']], feed_dict=feed_dict)
                
                    # update windows
                    elapsed_time = (time.time()-start_time) / step
                    self.update_windows(elapsed_time, results)
               
                    # log
                    msg = self.model.get_train_log(results['step'], self.time_window, self.loss_windows)
                    self.print_log(msg)
                else:
                    _ = sess.run(fetches['cls'], feed_dict=feed_dict)
                    
            ####################################
            # Phase 3: train the whole network #
            ####################################
            else:
                feed_dict = {
                    self.opt['gamma']: self.arch['training']['gamma'],
                    self.opt['lambda']: self.arch['training']['lambda']
                    }

                nIterD = self.arch['training']['nIterD']

                sess.run(fetches['gan'])

                # Display progress when reached a certain frequency
                if (step+1) % self.arch['training']['log_freq'] == 0:
                    for _ in range(nIterD - 1):
                        _, results = sess.run([fetches['vae_no_step_update'], fetches['info']], feed_dict=feed_dict)
                    _, results = sess.run([fetches['vae_step_update'], fetches['info']], feed_dict=feed_dict)
                
                    # update windows
                    elapsed_time = (time.time()-start_time) / step
                    self.update_windows(elapsed_time, results)
               
                    # log
                    msg = self.model.get_train_log(results['step'], self.time_window, self.loss_windows)
                    self.print_log(msg)
                    
                else:
                    for _ in range(nIterD - 1):
                        _ = sess.run(fetches['vae_no_step_update'], feed_dict=feed_dict)
                    _ = sess.run(fetches['vae_step_update'], feed_dict=feed_dict)
