import logging
import os

import numpy as np
import tensorflow as tf
from trainer.base import Trainer
from util.wrapper import load

class VAEUGANTrainer(Trainer):
    '''
    Trainer for training VAEUGAN 
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
        optimizer = tf.train.AdamOptimizer(lr, b1, b2)

        var_list = tf.trainable_variables()
        sp_e_vars = [v for v in var_list if 'SP_Encoder' in v.name]
        sp_g_vars = [v for v in var_list if 'SP_Decoder' in v.name]
        sp_d_vars = [v for v in var_list if 'SP_Discriminator' in v.name]

        r_sp = tf.placeholder(shape=[], dtype=tf.float32)
        k = tf.constant(self.arch['training']['clamping'], shape=[])
        self.loss['dis_sp2sp'] /= k

        obj_sp_Ez = self.loss['D_KL'] - self.loss['recon']
        obj_sp_Gx = r_sp * self.loss['dis_sp2sp'] - self.loss['recon']
        obj_sp_d = - self.loss['dis_sp2sp'] * k

        with tf.name_scope('Update'):
            opt_sp_d = optimizer.minimize(obj_sp_d, var_list=sp_d_vars)
            opt_sp_ds = [opt_sp_d]

            logging.info('The following variables are clamped:')
            with tf.control_dependencies(opt_sp_ds):
                with tf.name_scope('Clamping'):
                    for v in sp_d_vars:
                        v_clamped = tf.clip_by_value(v, -k, k)
                        clamping = tf.assign(v, v_clamped)
                        opt_sp_ds.append(clamping)
                        logging.info(v.name)

            opt_sp_g = optimizer.minimize(obj_sp_Gx, var_list=sp_g_vars, global_step=global_step)
            opt_sp_e = optimizer.minimize(obj_sp_Ez, var_list=sp_e_vars)

        return {
            'opt_d': opt_sp_ds,
            'opt_g': opt_sp_g,
            'opt_e': opt_sp_e,
            'gamma_sp': r_sp,
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
                    self.opt['gamma_sp']: 0.}

                # Display progress when reached a certain frequency
                if (step+1) % self.arch['training']['log_freq'] == 0:

                    _, results = sess.run([fetches['vae'], fetches['info']], feed_dict=feed_dict)
                    msg = self.model.get_train_log(results)
                    self.print_log(msg)

                    '''
                    # validation
                    valid_loss_all = []
                    for _ in range(self.valid['num_files']):
                        results = sess.run(fetches['valid'])
                        valid_loss_all.append(results)
                    
                    msg = self.model.get_valid_log(results['step'], valid_loss_all)
                    self.print_log(msg)
                    '''

                else:
                    _ = sess.run(fetches['vae'], feed_dict=feed_dict)

            else:
                feed_dict = {
                    self.opt['gamma_sp']: self.arch['training']['gamma_sp']}

                for _ in range(self.arch['training']['nIterD']):
                    sess.run(fetches['gan'])

                # Display progress when reached a certain frequency
                if (step+1) % self.arch['training']['log_freq'] == 0:

                    _, results = sess.run([fetches['vae'], fetches['info']], feed_dict=feed_dict)
                    msg = self.model.get_train_log(results)
                    self.print_log(msg)
                    
                    '''
                    # validation
                    valid_loss_all = []
                    for _ in range(self.valid['num_files']):
                        results = sess.run(fetches['valid'])
                        valid_loss_all.append(results)
                    
                    msg = self.model.get_valid_log(results['step'], valid_loss_all)
                    self.print_log(msg)
                    '''

                else:
                    _ = sess.run(fetches['vae'], feed_dict=feed_dict)