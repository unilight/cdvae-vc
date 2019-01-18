import tensorflow as tf
import numpy as np
import logging, os

from util.wrapper import load 

class Trainer(object):
    def __init__(self, model, train_data, valid_data, arch, args, dirs, ckpt):
        self.model = model
        self.loss = self.model.loss(train_data)
        self.valid = self.model.validate(valid_data)
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
        self.saver = tf.train.Saver()
        
        # define scaffold
        self.scaffold = tf.train.Scaffold(saver = self.saver)

        # define hooks
        hooks = self.get_hooks()

        # Initialize TensorFlow monitored training session
        session_creator = tf.train.ChiefSessionCreator(
                                scaffold = self.scaffold,
                                checkpoint_dir = self.dirs,
                                config = self.sess_config,
                                )
        self.sess =  tf.train.MonitoredSession(
                        session_creator=session_creator,
                        hooks = hooks,
                        )

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

    def get_hooks(self):
        # self.merged_summary_op = tf.summary.merge_all()
       
        saver_hook = tf.train.CheckpointSaverHook(
                checkpoint_dir = self.dirs, 
                save_steps = self.arch['training']['save_freq'],
                saver = self.saver,
                )

        # use scaffold to get summary op
        summary_hook = tf.train.SummarySaverHook(
                save_steps = self.arch['training']['summary_freq'],
                scaffold = self.scaffold,
                # summary_op = self.merged_summary_op,
                output_dir = self.dirs
                )

        stop_hook = tf.train.StopAtStepHook(
                last_step = self.arch['training']['max_iter']
                )

        step_counter_hook = tf.train.StepCounterHook(
                output_dir = self.dirs,
                every_n_steps = self.arch['training']['summary_freq']
                )

        return [saver_hook, summary_hook, stop_hook, step_counter_hook]
    
    def print_log(self, msg):
        print(msg)
        logging.info(msg)

    def restore(self):
        if self.ckpt:
            load(self.saver, self.sess, self.dirs, self.ckpt)
        elif self.args.logdir:
            load(self.saver, self.sess, self.dirs)
