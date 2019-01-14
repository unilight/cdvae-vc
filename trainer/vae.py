import logging
import os

import numpy as np
import tensorflow as tf
from trainer.base import Trainer

class VAETrainer(Trainer):
    '''
    Trainer for training VAE-style models 
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

        with tf.name_scope('Update'):
            opt = optimizer.minimize(self.loss['all'], var_list=var_list, global_step=global_step)
        return {
            'opt': opt,
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
            
            # Display progress when reached a certain frequency
            if (step+1) % self.arch['training']['log_freq'] == 0:
                results = sess.run(fetches['info'])
                msg = self.model.get_train_log(results)
                self.print_log(msg)

                # validation
                valid_loss_all = []
                for _ in range(self.valid['num_files']):
                    results = sess.run(fetches['valid'])
                    valid_loss_all.append(results)
                
                msg = self.model.get_valid_log(results['step'], valid_loss_all)
                self.print_log(msg)

            else:
                _ = sess.run(fetches['update'])
