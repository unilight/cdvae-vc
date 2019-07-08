#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Convert FEATURES with trained models.
# By Wen-Chin Huang 2019.06

import json
import os

import tensorflow as tf
import numpy as np

from importlib import import_module

import argparse
import logging

import sys
from preprocessing.vcc2018.feature_reader import Whole_feature_reader
from util.normalizer import MinMaxScaler, StandardScaler
from util.misc import read_hdf5, read_txt, load, get_default_logdir_output

def main():
    
    parser = argparse.ArgumentParser(
        description="Conversion.")
    parser.add_argument(
        "--logdir", required=True, type=str,
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
    parser.add_argument(
        "--type", default='test', type=str,
        help="test or valid (default is test)")
    
    
    parser.add_argument(
        "--input_feat", required=True, 
        type=str, help="input feature type")
    parser.add_argument(
        "--output_feat", required=True, 
        type=str, help="output feature type")
    parser.add_argument(
        "--mcd", action='store_true',
        help="calculate mcd or not")
    parser.add_argument(
        "--syn", action='store_true',
        help="synthesize voice or not")
    args = parser.parse_args()

    # make exp directory
    output_dir = get_default_logdir_output(args)
    tf.gfile.MakeDirs(output_dir)

    # set log level
    fmt = '%(asctime)s %(message)s'
    datefmt = '%m/%d/%Y %I:%M:%S'
    logFormatter = logging.Formatter(fmt, datefmt=datefmt)
    logging.basicConfig(
        level=logging.INFO,
        filename=os.path.join(output_dir, 'exp.log'),
        format=fmt,
        datefmt=datefmt,
        )
    consoleHandler = logging.StreamHandler()
    consoleHandler.setFormatter(logFormatter)
    logging.getLogger().addHandler(consoleHandler)
    logging.info('====================')
    logging.info('Conversion start')
    logging.info(args)

    # Load architecture
    arch = tf.gfile.Glob(os.path.join(args.logdir, 'architecture*.json'))[0]  # should only be 1 file
    with open(arch) as fp:
        arch = json.load(fp)
    
    # Load the model module
    module = import_module(arch['model_module'], package=None)
    MODEL = getattr(module, arch['model'])

    input_feat = args.input_feat
    input_feat_dim = arch['feat_param']['dim'][input_feat]
    output_feat = args.output_feat
    
    # read speakers
    spk_list = read_txt(arch['spklist'])

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

    # Define placeholders
    x_pl = tf.placeholder(tf.float32, [None, input_feat_dim])
    
    yh_pl = tf.placeholder(dtype=tf.int64, shape=[1,])
    yh = yh_pl * tf.ones(shape=[tf.shape(x_pl)[0],], dtype=tf.int64)
    yh = tf.expand_dims(yh, 0)
    
    # Define model
    model = MODEL(arch, normalizers)
    z, _ = model.encode(x_pl, input_feat)
    xh = model.decode(z, yh, output_feat)
    
    # make directories for output
    tf.gfile.MakeDirs(os.path.join(output_dir, 'latent'))
    tf.gfile.MakeDirs(os.path.join(output_dir, 'converted-{}'.format(output_feat)))
    
    # Define session
    with tf.Session() as sess:
    
        # define saver
        saver = tf.train.Saver()
        
        # load checkpoint
        if args.checkpoint is None:
            load(saver, sess, args.logdir,)
        else:
            _, ckpt = os.path.split(args.checkpoint)
            load(saver, sess, args.logdir, ckpt=ckpt)

        # get feature list, either validation set or test set
        if args.type == 'test':
            files = tf.gfile.Glob(arch['conversion']['test_file_pattern'].format(args.src))
        elif args.type == 'valid':
            files = []
            for p in arch['training']['valid_file_pattern']:
                files.extend(tf.gfile.Glob(p.replace('*', args.src)))
        files = sorted(files)

        # conversion
        for f in files:
            basename = os.path.split(f)[-1]
            path_to_latent = os.path.join(output_dir, 'latent', '{}-{}-{}'.format(args.src, args.trg, basename))
            path_to_cvt = os.path.join(output_dir, 'converted-{}'.format(output_feat), '{}-{}-{}'.format(args.src, args.trg, basename))
            logging.info(basename)

            # load source features
            src_data = Whole_feature_reader(f, arch['feat_param'])

            # 
            latent, cvt = sess.run([z, xh],
                               feed_dict={yh_pl : np.asarray([spk_list.index(args.trg)]),
                                          x_pl : src_data[input_feat] }
                              )
            # save bin
            with open(path_to_latent, 'wb') as fp:
                fp.write(latent.tostring())
            with open(path_to_cvt, 'wb') as fp:
                fp.write(cvt.tostring())
    
    # optionally calculate MCD
    if args.mcd:
        cmd = "python ./mcd_calculate.py" + \
                    " --type " + args.type + \
                    " --logdir " + output_dir + \
                    " --input_feat " + input_feat + \
                    " --output_feat " + output_feat
        print(cmd)
        os.system(cmd)
    
    # optionally synthesize waveform
    if args.syn:
        cmd = "python ./synthesize.py" + \
                    " --type " + args.type + \
                    " --logdir " + output_dir + \
                    " --input_feat " + input_feat + \
                    " --output_feat " + output_feat
        print(cmd)
        os.system(cmd)

if __name__ == '__main__':
    main()
