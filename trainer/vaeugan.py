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

        r = tf.placeholder(shape=[], dtype=tf.float32)
        k = tf.constant(self.arch['training']['clamping'], shape=[])
        self.loss['dis'] = self.loss['dis'] / k 

        obj_sp_Ez = self.loss['D_KL'] - self.loss['recon']
        obj_sp_Gx = r * self.loss['dis'] - self.loss['recon']
        obj_sp_d = - self.loss['dis'] * k 

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
            'gamma': r,
            'global_step': global_step
        }


    def print_log(self, result):
        msg = 'Iter {:05d}: '.format(result['step'])
        msg += 'recon = {:.4} '.format(result['recon'])
        msg += 'KL = {:.4} '.format(result['D_KL'])
        msg += 'Dis = {:.4} '.format(result['dis'])
        print(msg)
        logging.info(msg)

    def train(self):
        vae_saver = tf.train.Saver()
        run_metadata = tf.RunMetadata()
        merged_summary_op = tf.summary.merge_all()

        """ define fetches
            update_fetches: optimization only
            info_fetches: get logging infos
            valid_fetches: for validation
        """
        vae_fetches = {
            "opt_e": self.opt['opt_e'],
            "opt_g": self.opt['opt_g']
        }
        gan_fetches = {
            "opt_d": self.opt['opt_d']
        }
        info_fetches = {
            "D_KL": self.loss['D_KL'],
            "recon": self.loss['recon'],
            "dis": self.loss['dis'],
            "step": self.opt['global_step'],
        }
        valid_fetches = {
            "recon": self.valid['recon_sp'],
            "step": self.opt['global_step']
        }

        # define hooks
        saver = tf.train.Saver()
        saver_hook = tf.train.CheckpointSaverHook(
                checkpoint_dir=self.dirs, 
                save_steps = self.arch['training']['save_freq'],
                saver=saver
                )
        summary_hook = tf.train.SummarySaverHook(
                save_steps = self.arch['training']['summary_freq'],
                summary_op = merged_summary_op,
                output_dir = self.dirs
                )
        stop_hook = tf.train.StopAtStepHook(
                last_step = self.arch['training']['vae_iter'] + self.arch['training']['gan_iter']
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

            # load pretrained model
            if self.args.restore_path:
                load(saver, sess, self.args.restore_path)
            
            # Iterate through training steps
            while not sess.should_stop():

                # update global step
                step = tf.train.global_step(sess, self.opt['global_step'])
                
                if (step+1) <= self.arch['training']['vae_iter']:
                    feed_dict = {
                        self.opt['gamma']: 0.}

                    if (step+1) % self.arch['training']['log_freq'] == 0:
                        # Display progress when reached a certain frequency
                        _, results = sess.run([vae_fetches, info_fetches], feed_dict=feed_dict)
                        self.print_log(results)
                        '''
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
                        '''

                    else:
                        _ = sess.run(vae_fetches, feed_dict=feed_dict)

                else:
                    feed_dict = {
                        self.opt['gamma']: self.arch['training']['gamma']}

                    if (step+1) - self.arch['training']['vae_iter'] < 25 or (step+1) % 100 == 0:
                        nIterD = self.arch['training']['n_unroll_intense']
                    else:
                        nIterD = self.arch['training']['n_unroll']

                    for _ in range(nIterD):
                        sess.run(gan_fetches)

                    if (step+1) % self.arch['training']['log_freq'] == 0:
                        # Display progress when reached a certain frequency
                        _, results = sess.run([vae_fetches, info_fetches], feed_dict=feed_dict)
                        self.print_log(results)
                        '''
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
                        '''

                    else:
                        _ = sess.run(vae_fetches, feed_dict=feed_dict)