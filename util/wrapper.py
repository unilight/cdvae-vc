import json
import os
import sys
from datetime import datetime

import tensorflow as tf
import logging

def save(saver, sess, logdir, step):
    ''' Save a model to logdir/model.ckpt-[step] '''
    model_name = 'model.ckpt'
    checkpoint_path = os.path.join(logdir, model_name)
    print('Storing checkpoint to {} ...'.format(logdir), end="")
    sys.stdout.flush()

    if not os.path.exists(logdir):
        os.makedirs(logdir)

    saver.save(sess, checkpoint_path, global_step=step)
    print(' Done.')


def load(saver, sess, logdir, ckpt=None):
    '''
    Try to load model form a dir (search for the newest checkpoint)
    '''
    if ckpt:
        ckpt = os.path.join(logdir, ckpt)
        global_step = int(ckpt.split('/')[-1].split('-')[-1])
        logging.info('  Global step: {}'.format(global_step))
        saver.restore(sess, ckpt)
        return global_step
    else:
        ckpt = tf.train.latest_checkpoint(logdir)
        if ckpt:
            logging.info('  Checkpoint found: {}'.format(ckpt))
            global_step = int(ckpt.split('/')[-1].split('-')[-1])
            logging.info('  Global step: {}'.format(global_step))
            saver.restore(sess, ckpt)
            return global_step
        else:
            print('No checkpoint found')
            return None

def get_default_logdir_train(logdir_root='logdir'):
    STARTED_DATESTRING = datetime.now().strftime('%0m%0d-%0H%0M-%0S-%Y')
    logdir = os.path.join(logdir_root, STARTED_DATESTRING)
    print('Using default logdir: {}'.format(logdir))        
    return logdir

def get_default_logdir_output(args, conversion = True):
    STARTED_DATESTRING = datetime.now().strftime('%0m%0d-%0H%0M-%0S-%Y')
    if conversion:
        logdir = os.path.join(args.logdir, STARTED_DATESTRING+'-{}-{}'.format(args.src, args.trg))
    else:
        logdir = os.path.join(args.logdir, STARTED_DATESTRING+'-latent-codes')
    print('Logdir: {}'.format(logdir))        
    return logdir
