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
    args = parser.parse_args()

    #################################################################################

    # Make log directory
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

    with open(os.path.join(logdir, os.path.split(args.architecture)[-1]), 'w') as f:
        json.dump(arch, f, indent=4)
    
    # Load the model and trainer modules
    module = import_module(arch['model_module'], package=None)
    MODEL = getattr(module, arch['model'])

    module = import_module(arch['trainer_module'], package=None)
    TRAINER = getattr(module, arch['trainer'])
    
    # Load training data
    train_data = Frame_feature_reader(
        file_pattern = arch['training']['train_file_pattern'],
        feat_param = arch['feat_param'],
        batch_size = arch['training']['batch_size'],
    )
    valid_data = Whole_feature_reader_tf(
        file_pattern = arch['training']['valid_file_pattern'],
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
            for data in [train_data, valid_data]:
                data[k] = normalizer.forward_process(data[k])

        for data in [train_data, valid_data]:
            data[k] = tf.expand_dims(tf.expand_dims(data[k], 1), -1)
    
    # Load model and trainer
    model = MODEL(arch)
    loss = model.loss(train_data)
    valid = model.validate(valid_data)
    trainer = TRAINER(loss, valid, arch, args, logdir)
    
    # Train the model
    trainer.train()


if __name__ == '__main__':
    main()
