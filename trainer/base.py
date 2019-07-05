import tensorflow as tf
import numpy as np
import logging, os

from util.wrapper import load 
from util.wrapper import ValueWindow

class Trainer(object):
    def __init__(self, model, train_data, arch, args, dirs, ckpt):
        self.model = model
        self.loss = self.model.loss(train_data)
        self.arch = arch
        self.args = args
        self.dirs = dirs
        self.ckpt = ckpt
        
        # get optimization ops
        self.opt = self._optimize()

        # get metadata, and session configs for GPU
        self.sess_config = self._sess_config()
        self.run_metadata = tf.RunMetadata()
       
        # define saver
        self.saver = tf.train.Saver(max_to_keep=None)
        
        # define hooks
        hooks = self.get_hooks(self.saver)

        # Initialize TensorFlow monitored training session
        self.sess =  tf.train.MonitoredTrainingSession(
                        hooks = hooks,
                        config = self.sess_config,
                        )

        # init windows for time and each losses
        self.reset_time_windows()

    def reset_time_windows(self):
        self.time_window = ValueWindow(100)
        self.loss_windows = {}
        for k, v in self.loss.items():
            self.loss_windows[k] = ValueWindow(100)
        return


    def _optimize(self):
        """ To be implemented by child class
            Should rovide the following operators:
            opt: update operator
            global_step: global step
        """
        return {}
    
    def _sess_config(self):
        return tf.ConfigProto(
            allow_soft_placement=True,
            gpu_options=tf.GPUOptions(allow_growth=True))

    def get_hooks(self, saver):
        merged_summary_op = tf.summary.merge_all()
        
        saver_hook = tf.train.CheckpointSaverHook(
                checkpoint_dir = self.dirs, 
                save_steps = self.arch['training']['save_freq'],
                saver = saver,
                )
        summary_hook = tf.train.SummarySaverHook(
                save_steps = self.arch['training']['summary_freq'],
                summary_op = merged_summary_op,
                output_dir = self.dirs
                )
        stop_hook = tf.train.StopAtStepHook(
                last_step = self.arch['training']['max_iter']
                )

        return [saver_hook, summary_hook, stop_hook]
    
    def print_log(self, msg):
        print(msg)
        logging.info(msg)

    def restore(self):
        if self.ckpt:
            load(self.saver, self.sess, self.dirs, self.ckpt)
        elif self.args.logdir:
            load(self.saver, self.sess, self.dirs)

    def update_windows(self, elapsed_time, results):
        self.time_window.append(elapsed_time)
        for k in self.loss_windows:
            self.loss_windows[k].append(results[k])
        
        
