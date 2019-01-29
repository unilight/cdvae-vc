import logging
import os

import numpy as np
import tensorflow as tf
from trainer.base import Trainer

class VAESGANTrainer(Trainer):
    '''
    Trainer for training VAE-style models 
    '''

    def _optimize(self):
        """ get the following operators:
            opt: update operator
            global_step: global step
        """
        
        trainables = tf.trainable_variables()
        g_vars = [v for v in trainables if 'Decoder' in v.name]
        d_vars = [v for v in trainables if 'Discriminator' in v.name]
        e_vars = [v for v in trainables if 'Encoder' in v.name]
        
        global_step = tf.train.get_or_create_global_step()
        lr = self.arch['training']['lr']
        b1 = self.arch['training']['beta1']
        b2 = self.arch['training']['beta2']
        optimizer_d = tf.train.AdamOptimizer(lr, b1, b2)
        optimizer_g = tf.train.AdamOptimizer(lr, b1, b2)
        
        r = tf.placeholder(shape=[], dtype=tf.float32)
        gp_weight = self.arch['training']['gp_weight']
        
        obj_Ez = self.loss['D_KL'] - self.loss['recon']
        obj_Gx = -r * self.loss['wgan'] - self.loss['recon']
        obj_D = self.loss['wgan'] + gp_weight * self.loss['wgan_gp']   # discriminator loss
        
        opt_d = optimizer_d.minimize(obj_D, var_list=d_vars)
        opt_ds = [opt_d]

        opt_g = optimizer_g.minimize(obj_Gx, var_list=g_vars, global_step=global_step)
        opt_e = optimizer_g.minimize(obj_Ez, var_list=e_vars)

        return {
            'opt_d': opt_ds,
            'opt_g': opt_g,
            'opt_e': opt_e,
            'gamma': r,
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
                    self.opt['gamma']: 0.}

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
                    self.opt['gamma']: self.arch['training']['gamma']}

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

