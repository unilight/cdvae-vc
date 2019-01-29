import os
import json

import tensorflow as tf
import numpy as np

from util.wrapper import save, get_default_logdir_train
from importlib import import_module

import argparse
import logging

import sys
from preprocessing.vcc2018.feature_reader import Frame_feature_reader, Whole_feature_reader_tf
from preprocessing.normalizer import MinMaxScaler
from preprocessing.utils import read_hdf5

def main():
    
    parser = argparse.ArgumentParser(
        description="train the model.")
    parser.add_argument(
        "--architecture", default='architectures/architecture-cdvae.json', type=str,
        help="network architecture")
    parser.add_argument(
        "--logdir", default=None, type=str,
        help="path of log directory")
    parser.add_argument(
        "--checkpoint", default=None, type=str,
        help="path of checkpoint")
    parser.add_argument(
        "--src", default=None, required=True, type=str,
        help="source speaker")
    parser.add_argument(
        "--trg", default=None, required=True, type=str,
        help="target speaker")
    
    args = parser.parse_args()

    #################################################################################

    # check logdir and checkpoint and make log directory if necessary
    if args.checkpoint:
        logdir, ckpt = os.path.split(args.checkpoint)
    else:
        ckpt = None
        if args.logdir:
            logdir = args.logdir
        else:
            logdir = get_default_logdir_train()
            tf.gfile.MakeDirs(logdir)

    
    # set log level
    logging.basicConfig(
        level=logging.INFO,
        filename=os.path.join(logdir, 'training.log'),
        format='%(asctime)s %(message)s',
        datefmt='%m/%d/%Y %I:%M:%S',
        )
    
    # Load network architecture and write to logdir
    with open(args.architecture) as f:
        arch = json.load(f)

    arch['src'] = args.src
    arch['trg'] = args.trg

    with open(os.path.join(logdir, os.path.split(args.architecture)[-1]), 'w') as f:
        json.dump(arch, f, indent=4)
    
    # Load the model and trainer modules
    module = import_module(arch['model_module'], package=None)
    MODEL = getattr(module, arch['model'])

    module = import_module(arch['trainer_module'], package=None)
    TRAINER = getattr(module, arch['trainer'])
    
    src = arch['src']
    trg = arch['trg']

    train_data_src_file_pattern = [file_pattern.format(src) for file_pattern in arch['training']['train_file_pattern']['src']]
    train_data_trg_file_pattern = [file_pattern.format(trg) for file_pattern in arch['training']['train_file_pattern']['trg']]
    valid_data_src_file_pattern = [file_pattern.format(src) for file_pattern in arch['training']['valid_file_pattern']['src']]


    # Load training data
    train_data_src = Frame_feature_reader(
        file_pattern = train_data_src_file_pattern,
        feat_param = arch['feat_param'],
        batch_size = arch['training']['batch_size'],
    )
    train_data_trg = Frame_feature_reader(
        file_pattern = train_data_trg_file_pattern,
        feat_param = arch['feat_param'],
        batch_size = arch['training']['batch_size'],
    )
    valid_data_src = Whole_feature_reader_tf(
        file_pattern = valid_data_src_file_pattern,
        feat_param = arch['feat_param'],
        num_epochs=None
    )

    # Load statistics, normalize and NCHW
    normalizers = {}
    for k in arch['normalizer_files']:
        if (arch['normalizer_files'][k]['max'] is not None
            and arch['normalizer_files'][k]['max'] is not None):
            normalizer = MinMaxScaler(
                xmax=np.fromfile(os.path.join(arch['stat_dir'], arch['normalizer_files'][k]['max'])),
                xmin=np.fromfile(os.path.join(arch['stat_dir'], arch['normalizer_files'][k]['min'])),
            )
            normalizers[k] = normalizer
            for data in [train_data_src, train_data_trg, valid_data_src]:
                data[k] = normalizer.forward_process(data[k])

        for data in [train_data_src, train_data_trg, valid_data_src]:
            data[k] = tf.expand_dims(tf.expand_dims(data[k], 1), -1)
    
    # Load model and trainer
    model = MODEL(arch, is_training = True)
    trainer = TRAINER(model,
                      [train_data_src, train_data_trg],
                      [valid_data_src],
                      arch, args, logdir, ckpt)
    
    # Train the model
    trainer.train()


if __name__ == '__main__':
    main()
