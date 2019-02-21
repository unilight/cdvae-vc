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
    
    for i, (bin_path, anasyn_path, feat_path) in enumerate(file_list):
        input_feat = arch['conversion']['input']
        input_feat_dim = arch['feat_param']['dim'][input_feat]
        output_feat = arch['conversion']['output']
        
        # define paths
        output_dir = os.path.dirname(bin_path).replace(output_feat, 'wav')
        basename = os.path.splitext(os.path.split(bin_path)[-1])[0]
        wav_name = os.path.join(output_dir, basename + '.wav')
        gv_wav_name = os.path.join(output_dir, basename + '-gv.wav')
        plant_wav_name = os.path.join(output_dir, basename + '-plant.wav')

        # read source features , anasyn features and converted spectral features
        src_data = Whole_feature_reader(feat_path, arch['feat_param'])
        cvt = np.fromfile(bin_path, dtype = np.float32).reshape([-1, input_feat_dim])
        anasyn = np.fromfile(anasyn_path, dtype = np.float32).reshape([-1, input_feat_dim])

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

        ##########################################################

        # spectral planting
        en_anasyn = np.c_[src_data['en_mcc'], anasyn]
        
        """
        ## get sp of anasyn and cvt
        sp_cvt = pysptk.mc2sp(cvt, arch['feat_param']['mcep_alpha'], arch['feat_param']['fftl'])
        sp_anasyn = pysptk.mc2sp(anasyn, arch['feat_param']['mcep_alpha'], arch['feat_param']['fftl'])

        ## spectral gain
        gain = sp_cvt / sp_anasyn

        ## planting
        src_idx = np.where(src_data['f0']>0)[0]
        sp_h = np.expand_dims(src_data['en_sp'], 1) * np.power(10., src_data['sp'].copy())
        sp_h[src_idx] = sp_h[src_idx] * gain[src_idx]
        
        """
        
        ## gain
        gain = cvt - anasyn

        ## planting
        src_idx = np.where(src_data['f0']>0)[0]
        mcc_h = src_data['mcc'].copy()
        mcc_h[src_idx] = mcc_h[src_idx] + gain[src_idx]
        mcc_h = np.c_[src_data['en_mcc'], mcc_h]


        world_synthesis(plant_wav_name, arch['feat_param'],
                        lf0.astype(np.float64).copy(order='C'),
                        src_data['ap'].astype(np.float64).copy(order='C'),
                        #sp_h.astype(np.float64).copy(order='C'),
                        #'sp')
                        mcc_h.astype(np.float64).copy(order='C'),
                        'mcc')



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
    anasyn_list = sorted(tf.gfile.Glob(os.path.join(args.logdir, 'anasyn-{}'.format(output_feat), '*.bin')))
    feat_list = sorted(tf.gfile.Glob(arch['conversion']['test_file_pattern'].format(src)))
    assert(len(bin_list) == len(feat_list))
    file_list = list(zip(bin_list, anasyn_list, feat_list))
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
