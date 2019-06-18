#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Train a VC model.
# By Wen-Chin Huang 2019.06


import os
import json

import tensorflow as tf
import numpy as np

from util.wrapper import save, get_default_logdir_train
from importlib import import_module

import argparse
import logging

import sys
from preprocessing.vcc2018.feature_reader import  Segment_feature_reader
from preprocessing.normalizer import MinMaxScaler
from preprocessing.utils import read_hdf5

def main():
    
    parser = argparse.ArgumentParser(
        description="Train the model.")
    parser.add_argument(
        "--architecture", required=True, type=str,
        help="network architecture")
    parser.add_argument(
        "--note", required=True, type=str,
        help="note on experiemnt")
    parser.add_argument(
        "--logdir", default=None, type=str,
        help="path of log directory")
    parser.add_argument(
        "--checkpoint", default=None, type=str,
        help="path of checkpoint")
    parser.add_argument(
        "--seed", default=12,
        type=int, help="initialization seed")
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
            logdir = get_default_logdir_train(args.note)
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
    train_data = Segment_feature_reader(
        file_pattern = arch['training']['train_file_pattern'],
        feat_param = arch['feat_param'],
        batch_size = arch['training']['batch_size'],
        crop_length = arch['training']['crop_length'],
    )

    # Load statistics, normalize and NCHW
    normalizers = {}
    for k in arch['normalizer']:
        normalizers[k] = {}
        for norm_type in arch['normalizer'][k]['type']:
            if norm_type == 'minmax':
                normalizer = MinMaxScaler(
                    xmax=read_hdf5(arch['stats'], '/max/' + k),
                    xmin=read_hdf5(arch['stats'], '/min/' + k),
                )
            elif norm_type == 'meanvar':
                normalizer = StandardScaler(
                    mu=read_hdf5(arch['stats'], '/mean/' + k),
                    std=read_hdf5(arch['stats'], '/scale/' + k),
                )

            normalizers[k][norm_type] = normalizer

    # set random seed
    tf.set_random_seed(args.seed)

    # Load model and trainer
    model = MODEL(arch, normalizers = normalizers)
    trainer = TRAINER(model, train_data, arch, args, logdir, ckpt)
    
    # Train the model
    trainer.train()


if __name__ == '__main__':
    main()
