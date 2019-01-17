import json
import os

import tensorflow as tf
import numpy as np

from util.wrapper import load
from datetime import datetime
from importlib import import_module

import pysptk
import pyworld as pw
from scipy.io import loadmat, savemat
from scipy.io import wavfile

import argparse
import logging
import multiprocessing as mp

import sys
from preprocessing.vcc2018.feature_reader import Whole_feature_reader, world_synthesis
from preprocessing.normalizer import MinMaxScaler
from preprocessing.utils import read_hdf5, read_txt
from preprocessing.postfilter import fast_MLGV
from preprocessing.f0transformation import log_linear_transformation

def read_and_synthesize(file_list, arch, stats, ):
    
    for i, (bin_path, feat_path) in enumerate(file_list):
        input_feat = arch['conversion']['input']
        input_feat_dim = arch['feat_param']['dim'][input_feat]
        output_feat = arch['conversion']['output']
        
        # define paths
        output_dir = os.path.dirname(bin_path).replace(output_feat, 'wav')
        basename = os.path.splitext(os.path.split(bin_path)[-1])[0]
        wav_name = os.path.join(output_dir, basename + '.wav')
        gv_wav_name = os.path.join(output_dir, basename + '-gv.wav')

        # read source features and converted spectral features
        src_data = Whole_feature_reader(feat_path, arch['feat_param'])
        cvt = np.fromfile(bin_path, dtype = np.float32).reshape([-1, input_feat_dim])

        # f0 conversion
        lf0 = log_linear_transformation(src_data['f0'], stats)

        # apply gv post filtering to converted
        cvt_gv = fast_MLGV(cvt, stats['gv_t'])
        
        # energy compensation
        if output_feat == 'mcc':
            en_cvt = np.c_[src_data['en_mcc'], cvt]
            en_cvt_gv = np.c_[src_data['en_mcc'], cvt_gv]
        elif output_feat == 'sp':
            cvt = np.power(10., cvt)
            en_cvt = np.expand_dims(src_data['en_sp'], 1) * cvt
            cvt_gv = np.power(10., cvt_gv)
            en_cvt_gv = np.expand_dims(src_data['en_sp'], 1) * cvt_gv

        # synthesis
        world_synthesis(wav_name, arch['feat_param'],
                        lf0.astype(np.float64).copy(order='C'),
                        src_data['ap'].astype(np.float64).copy(order='C'),
                        en_cvt.astype(np.float64).copy(order='C'),
                        output_feat)
        world_synthesis(gv_wav_name, arch['feat_param'],
                        lf0.astype(np.float64).copy(order='C'),
                        src_data['ap'].astype(np.float64).copy(order='C'),
                        en_cvt_gv.astype(np.float64).copy(order='C'),
                        output_feat)


def main():
    
    parser = argparse.ArgumentParser(
        description="synthesize waveforms using converted files.")
    parser.add_argument(
        "--logdir", required=True, type=str,
        help="path of log directory")
    parser.add_argument(
        "--n_jobs", default=12,
        type=int, help="number of parallel jobs")
    args = parser.parse_args()
   
    # set log level
    fmt = '%(asctime)s %(message)s'
    datefmt = '%m/%d/%Y %I:%M:%S'
    logFormatter = logging.Formatter(fmt, datefmt=datefmt)
    logging.basicConfig(
        level=logging.INFO,
        filename=os.path.join(args.logdir, 'exp.log'),
        format=fmt,
        datefmt=datefmt,
        )
    consoleHandler = logging.StreamHandler()
    consoleHandler.setFormatter(logFormatter)
    logging.getLogger().addHandler(consoleHandler)
    logging.info('====================')
    logging.info('Synthesize start')
    logging.info(args)

    train_dir = os.sep.join(args.logdir.split(os.sep)[:-2])
    output_dir = args.logdir.split(os.sep)[-2]
    src, trg = output_dir.split('-')[-2:]
    
    # Load architecture
    arch = tf.gfile.Glob(os.path.join(train_dir, 'architecture*.json'))[0]  # should only be 1 file
    with open(arch) as fp:
        arch = json.load(fp)
    
    input_feat = arch['conversion']['input']
    output_feat = arch['conversion']['output']
    
    # Load statistics
    mu_s, std_s = np.fromfile(os.path.join(arch['stat_dir'], '{}.npf'.format(src)), np.float32)
    mu_t, std_t = np.fromfile(os.path.join(arch['stat_dir'], '{}.npf'.format(trg)), np.float32)
    gv_t = loadmat(os.path.join(arch['stat_dir'], 'gv_{}'.format(input_feat), '{}.mat'.format(trg)))['gv_{}'.format(output_feat)].flatten()
    stats = {
        'mu_s' : mu_s, 'std_s' : std_s,
        'mu_t' : mu_t, 'std_t' : std_t,
        'gv_t' : gv_t
    }

    # Make directory
    tf.gfile.MakeDirs(os.path.join(args.logdir, 'converted-wav'))

    # Get and divide list
    bin_list = sorted(tf.gfile.Glob(os.path.join(args.logdir, 'converted-{}'.format(output_feat), '*.bin')))
    feat_list = sorted(tf.gfile.Glob(arch['conversion']['test_file_pattern'].format(src)))
    assert(len(bin_list) == len(feat_list))
    file_list = list(zip(bin_list, feat_list))
    logging.info("number of utterances = %d" % len(file_list))
    file_lists = np.array_split(file_list, args.n_jobs)
    file_lists = [f_list.tolist() for f_list in file_lists]

    # multi processing
    processes = []
    for f in file_lists:
        p = mp.Process(target=read_and_synthesize, args=(f, arch, stats,))
        p.start()
        processes.append(p)

    # wait for all process
    for p in processes:
        p.join()

if __name__ == '__main__':
    main()
