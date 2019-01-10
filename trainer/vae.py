import logging
import os

import numpy as np
import tensorflow as tf
from trainer.base import Trainer

class VAETrainer(Trainer):
    '''
    Trainer for training VAE 
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


    def print_log(self, result):
        msg = 'Iter {:05d}: '.format(result['step'])
        msg += 'recon = {:.4} '.format(result['recon'])
        msg += 'KL = {:.4} '.format(result['D_KL'])
        print(msg)
        logging.info(msg)

    def train(self):
        run_metadata = tf.RunMetadata()
        merged_summary_op = tf.summary.merge_all()

        """ define fetches
            update_fetches: get logging infos
            info_fetches: optimization only
            valid_fetches: for validation
        """
        update_fetches = self.opt['opt']
        info_fetches = {
            "D_KL": self.loss['D_KL'],
            "recon": self.loss['recon'],
            "opt": self.opt['opt'],
            "step": self.opt['global_step'],
        }
        valid_fetches = {
            "recon": self.valid['recon_sp'],
            "step": self.opt['global_step'],
        }

        # define hooks
        saver_hook = tf.train.CheckpointSaverHook(
                checkpoint_dir=self.dirs, 
                save_steps = self.arch['training']['save_freq']
                )
        summary_hook = tf.train.SummarySaverHook(
                save_steps = self.arch['training']['summary_freq'],
                summary_op = merged_summary_op,
                )
        stop_hook = tf.train.StopAtStepHook(
                last_step = self.arch['training']['max_iter']
                )
        
        # Define settings for gpu
        sess_config = tf.ConfigProto(
            allow_soft_placement=True,
            gpu_options=tf.GPUOptions(allow_growth=True))

        # Initialize TensorFlow monitored training session
        with tf.train.MonitoredTrainingSession(
                hooks=[saver_hook, summary_hook, stop_hook],
                config = sess_config,
                ) as sess:
            
            # Iterate through training steps
            while not sess.should_stop():

                # update global step
                step = tf.train.global_step(sess, self.opt['global_step'])
                
                # Display progress when reached a certain frequency
                if (step+1) % self.arch['training']['log_freq'] == 0:
                    results = sess.run(info_fetches)
                    self.print_log(results)

                    # validation
                    valid_loss_all = []
                    for _ in range(self.valid['num_files']):
                        results = sess.run(valid_fetches)
                        valid_loss_all.append(results['recon'])
                    
                    valid_loss_avg = np.mean(np.array(valid_loss_all))
                    msg = 'Validation in Iter {:05d}: '.format(results['step'])
                    msg += 'recon = {:.4} '.format(valid_loss_avg)
                    print(msg)
                    logging.info(msg)

                else:
                    _ = sess.run(update_fetches)
