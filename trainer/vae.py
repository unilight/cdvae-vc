import logging
import os

import numpy as np
import tensorflow as tf
from trainer.base import Trainer
from util.misc import ValueWindow
import time

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
            
            # Display progress when reached a certain frequency
            if (step+1) % self.arch['training']['log_freq'] == 0:
                results = sess.run(fetches['info'])

                # update windows
                elapsed_time = (time.time()-start_time) / step
                self.update_windows(elapsed_time, results)

                # log 
                msg = self.model.get_train_log(results['step'], self.time_window, self.loss_windows)
                self.print_log(msg)

            else:
                _ = sess.run(fetches['update'])
