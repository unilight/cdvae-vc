import json
import os

import tensorflow as tf
import numpy as np

import pysptk
import pyworld as pw
from scipy.io import wavfile
import scipy
from fastdtw import fastdtw

import argparse
import logging
import multiprocessing as mp

import sys
from preprocessing.vcc2018.feature_reader import Whole_feature_reader
from preprocessing.normalizer import MinMaxScaler

def read_and_synthesize(file_list, arch, MCD, ):
    
    for i, (bin_path, src_feat_path, trg_feat_path) in enumerate(file_list):
        input_feat = arch['conversion']['input']
        input_feat_dim = arch['feat_param']['dim'][input_feat]
        output_feat = arch['conversion']['output']
        basename = os.path.splitext(os.path.split(bin_path)[-1])[0]
        
        # read source features , target features and converted mcc
        src_data = Whole_feature_reader(src_feat_path, arch['feat_param'])
        trg_data = Whole_feature_reader(trg_feat_path, arch['feat_param'])
        cvt = np.fromfile(bin_path, dtype = np.float32).reshape([-1, arch['feat_param']['dim'][output_feat]])

        # Handle if not mcc
        if output_feat == 'sp':
            cvt = np.power(10., cvt)
            en_cvt = src_data['en_sp'] * cvt
            mcc_cvt = pysptk.sp2mc(en_cvt, arch['feat_param']['mcc_dim'], arch['feat_param']['mcep_alpha'])
            raise ValueError
        elif output_feat == 'mcc':
            mcc_cvt = cvt
        else:
            logging.info('Currently do not support types other than mcc and sp.' )
            raise ValueError

        # non-silence parts
        trg_idx = np.where(trg_data['f0']>0)[0]
        trg_mcc = trg_data['mcc'][trg_idx]
        src_idx = np.where(src_data['f0']>0)[0]
        mcc_cvt = mcc_cvt[src_idx]

        # DTW
        _, path = fastdtw(mcc_cvt, trg_mcc, dist=scipy.spatial.distance.euclidean)
        twf = np.array(path).T
        cvt_mcc_dtw = mcc_cvt[twf[0]]
        trg_mcc_dtw = trg_mcc[twf[1]]

        # MCD 
        diff2sum = np.sum((cvt_mcc_dtw - trg_mcc_dtw)**2, 1)
        mcd = np.mean(10.0 / np.log(10.0) * np.sqrt(2 * diff2sum))
        logging.info(f'{basename} {mcd}')
        MCD.append(mcd)

def main():
    
    parser = argparse.ArgumentParser(
        description="calculate MCD.")
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
    logging.info('MCD calculation')
    logging.info(args)
    
    train_dir = os.sep.join(args.logdir.split(os.sep)[:-2])
    output_dir = args.logdir.split(os.sep)[-2]
    src, trg = output_dir.split('-')[-2:]
    
    # Load architecture
    arch = tf.gfile.Glob(os.path.join(train_dir, 'architecture*.json'))[0]  # should only be 1 file
    with open(arch) as fp:
        arch = json.load(fp)
    
    # Get and divide list
    bin_list = sorted(tf.gfile.Glob(os.path.join(args.logdir, 'converted-mcc', '*.bin')))
    src_feat_list = sorted(tf.gfile.Glob(arch['conversion']['test_file_pattern'].format(src)))
    trg_feat_list = sorted(tf.gfile.Glob(arch['conversion']['test_file_pattern'].format(trg)))
    assert(len(bin_list) == len(src_feat_list))
    file_list = list(zip(bin_list, src_feat_list, trg_feat_list))
    logging.info("number of utterances = %d" % len(file_list))
    file_lists = np.array_split(file_list, args.n_jobs)
    file_lists = [f_list.tolist() for f_list in file_lists]

    # multi processing
    with mp.Manager() as manager:
        MCD = manager.list()
        processes = []
        for f in file_lists:
            p = mp.Process(target=read_and_synthesize, args=(f, arch, MCD,))
            p.start()
            processes.append(p)

        # wait for all process
        for p in processes:
            p.join()

        mMCD = np.mean(np.array(MCD))
        logging.info(f'Mean MCD: {mMCD}')

if __name__ == '__main__':
    main()
