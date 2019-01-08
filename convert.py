import json
import os

import tensorflow as tf
import numpy as np

from datetime import datetime
from importlib import import_module

import argparse
import logging

import sys
from preprocessing.vcc2018.feature_reader import Whole_feature_reader
from preprocessing.normalizer import MinMaxScaler
from preprocessing.utils import read_hdf5, read_txt
from util.wrapper import load, get_default_logdir_output

def main():
    
    parser = argparse.ArgumentParser(
        description="convert files.")
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
    parser.add_argument('--stat_dir', type=str,
        default='/mnt/md1/datasets/vcc2018/world/etc-new',
        help='configuration directory')
    parser.add_argument('--file_pattern', type=str,
        default='/mnt/md1/datasets/vcc2018/world/bin-dynamic/no_VAD/ev/{}/*.bin',
        help='file pattern')
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

    input_feat = arch['conversion']['input']
    input_feat_dim = arch['feat_param']['dim'][input_feat]
    output_feat = arch['conversion']['output']
    
    # read speakers
    spk_list = read_txt(arch['spklist'])

    # Load statistics, normalize and NCHW
    normalizers = {}
    for k in arch['normalizer_files']:
        normalizer = MinMaxScaler(
            xmax=np.fromfile(os.path.join(arch['stat_dir'], arch['normalizer_files'][k]['max'])),
            xmin=np.fromfile(os.path.join(arch['stat_dir'], arch['normalizer_files'][k]['min'])),
        )
        normalizers[k] = normalizer

    # Define placeholders
    x_pl = tf.placeholder(tf.float32, [None, input_feat_dim])
    x = tf.expand_dims(tf.expand_dims(normalizers[input_feat].forward_process(x_pl), 1), -1)
    yh_pl = tf.placeholder(dtype=tf.int64, shape=[1,])
    yh = yh_pl * tf.ones(shape=[tf.shape(x)[0],], dtype=tf.int64)
    
    # Define model
    model = MODEL(arch)
    z = model.encode(x)
    xh = model.decode(z, yh)
    xh = tf.squeeze(xh)
    xh = normalizers[output_feat].backward_process(xh)
    
    # make directories for output
    tf.gfile.MakeDirs(os.path.join(output_dir, 'latent'))
    tf.gfile.MakeDirs(os.path.join(output_dir, f'converted-{output_feat}'))
    
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

        files = sorted(tf.gfile.Glob(arch['conversion']['test_file_pattern'].format(args.src)))
        for f in files:
            basename = os.path.split(f)[-1]
            path_to_latent = os.path.join(output_dir, 'latent', '{}-{}-{}'.format(args.src, args.trg, basename))
            path_to_cvt = os.path.join(output_dir, f'converted-{output_feat}', '{}-{}-{}'.format(args.src, args.trg, basename))
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

if __name__ == '__main__':
    main()
