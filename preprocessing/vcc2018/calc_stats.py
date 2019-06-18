#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Statistics calculation for VCC2018 dataset
# Including min/max [-1,+1] and mean/var (0,1), GV
# By Wen-Chin Huang 2019.06

import argparse
import logging

import numpy as np
import os

from sklearn.preprocessing import StandardScaler

import sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from utils import (find_files, read_txt, write_hdf5)
from vcc2018.feature_reader import Whole_feature_reader

def calc_stats(file_list, feat_param, spk_list, args):
    
    SP_DIM = feat_param['fftl'] // 2 + 1 
    MCC_DIM = feat_param['mcep_dim']
    FEAT_DIM = feat_param['feat_dim']
    
    standard_scaler = StandardScaler()

    sp_all = []
    mcc_all = []
    f0_all = []
    spk_all = []

    # process over all of data
    for i, filename in enumerate(file_list):
        logging.info("now processing %s (%d/%d)" % (filename, i + 1, len(file_list)))
        reader = Whole_feature_reader
        feats = reader(filename, feat_param)
        
        sp_all          .append(feats['sp'])
        mcc_all         .append(feats['mcc'])
        f0_all          .append(feats['f0'])
        spk_all         .append(feats['speaker'])

        # calculate mean/var  [sp, mcc, f0, mag_sgram, mel_sgram]
        global_feats = np.concatenate([feats['sp'], feats['mcc'], np.expand_dims(feats['f0'], 1)], axis=1)
        standard_scaler.partial_fit(global_feats)

    sp_all = np.concatenate(sp_all, axis=0)
    mcc_all = np.concatenate(mcc_all, axis=0)
    f0_all = np.concatenate(f0_all, axis=0)
    spk_all = np.concatenate(spk_all, axis=0)

    # F0, GV stats
    for s in spk_list:
        logging.info("Speaker %s" % (s))
        f0 = f0_all[spk_list.index(s) == spk_all]
        logging.info('  Number of frames: {}'.format(len(f0)))
        
        # Calculate mean and std of f0
        f0 = f0[f0 > 2.]
        f0 = np.log(f0)
        mu, std = f0.mean(), f0.std()
        write_hdf5(args.stats, "/f0/" + s + "/mean", np.float32(mu))
        write_hdf5(args.stats, "/f0/" + s + "/std", np.float32(std))
        
        # Calculate gv for mcc and sp, cov matrix of mcc
        sp = sp_all[spk_list.index(s) == spk_all]
        gv_sp = np.var(sp, axis=0)
        mcc = mcc_all[spk_list.index(s) == spk_all]
        gv_mcc = np.var(mcc, axis=0)
        cov_mcc = np.diag(np.var(mcc,axis=0))
        write_hdf5(args.stats, "/gv_sp/" + s, np.float32(gv_sp))
        write_hdf5(args.stats, "/gv_mcc/" + s, np.float32(gv_mcc))
        write_hdf5(args.stats, "/cov_mcc/" + s, np.float32(cov_mcc))

    # min-max stats
    logging.info("Min-Max stats")
    minimum_sp = np.percentile(sp_all, 0.5, axis=0)
    maximum_sp = np.percentile(sp_all, 99.5, axis=0)
    minimum_mcc = np.percentile(mcc_all, 0.5, axis=0)
    maximum_mcc = np.percentile(mcc_all, 99.5, axis=0)
    minimum_f0 = np.percentile(f0_all, 0.5, axis=0)
    maximum_f0 = np.percentile(f0_all, 99.5, axis=0)
    
    write_hdf5(args.stats, "/min/sp", np.float32(minimum_sp))
    write_hdf5(args.stats, "/max/sp", np.float32(maximum_sp))
    write_hdf5(args.stats, "/min/mcc", np.float32(minimum_mcc))
    write_hdf5(args.stats, "/max/mcc", np.float32(maximum_mcc))
    write_hdf5(args.stats, "/min/f0", np.float32(minimum_f0))
    write_hdf5(args.stats, "/max/f0", np.float32(maximum_f0))

    # mean-var stats
    logging.info("Mean-Var stats")
    mean            = standard_scaler.mean_
    scale           = standard_scaler.scale_
    mean_sp         = mean [:SP_DIM]
    scale_sp        = scale[:SP_DIM]
    mean_mcc        = mean [SP_DIM:SP_DIM+MCC_DIM]
    scale_mcc       = scale[SP_DIM:SP_DIM+MCC_DIM]
    mean_f0         = mean [SP_DIM+MCC_DIM]
    scale_f0        = scale[SP_DIM+MCC_DIM]


    write_hdf5(args.stats, "/mean/sp", np.float32(mean_sp))
    write_hdf5(args.stats, "/scale/sp", np.float32(scale_sp))
    write_hdf5(args.stats, "/mean/mcc", np.float32(mean_mcc))
    write_hdf5(args.stats, "/scale/mcc", np.float32(scale_mcc))
    write_hdf5(args.stats, "/mean/f0", np.float32(mean_f0))
    write_hdf5(args.stats, "/scale/f0", np.float32(scale_f0))
    
    logging.info("Calculation complete.")


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--bindir", required=True, type=str,
        help="name of the dir of bin files")
    parser.add_argument(
        "--stats", required=True, type=str,
        help="filename of hdf5 format")
    parser.add_argument(
        "--spklist", required=True, type=str,
        help="list of speakers")
    parser.add_argument(
        "--verbose", default=1,
        type=int, help="log message level")

    args = parser.parse_args()

    # set log level
    if args.verbose == 1:
        logging.basicConfig(level=logging.INFO,
                            format='%(asctime)s (%(module)s:%(lineno)d) %(levelname)s: %(message)s',
                            datefmt='%m/%d/%Y %I:%M:%S')
    elif args.verbose > 1:
        logging.basicConfig(level=logging.DEBUG,
                            format='%(asctime)s (%(module)s:%(lineno)d) %(levelname)s: %(message)s',
                            datefmt='%m/%d/%Y %I:%M:%S')
    else:
        logging.basicConfig(level=logging.WARN,
                            format='%(asctime)s (%(module)s:%(lineno)d) %(levelname)s: %(message)s',
                            datefmt='%m/%d/%Y %I:%M:%S')
        logging.warn("logging is disabled.")

    # show argmument
    for key, value in vars(args).items():
        logging.info("%s = %s" % (key, str(value)))
    
    # define feat param here
    feat_param = {
        'fs'                : 22050,
        'shift_ms'          : 5,
        'length_ms'         : 25,
        'fftl'              : 1024,
        'n_mels'            : 80,
        'mcep_dim'          : 34,
        'mcep_alpha'        : 0.455,
        'feat_dim'          : 1064,
    }
    
    # read speakers
    spk_list = read_txt(args.spklist)

    # read file list, for trainind data only
    file_list = sorted(find_files(args.bindir, "[12]*.bin"))
    logging.info("number of utterances = %d" % len(file_list))

    # calculate statistics
    if not os.path.exists(os.path.dirname(args.stats)):
        os.makedirs(os.path.dirname(args.stats))
    calc_stats(file_list, feat_param, spk_list, args)


if __name__ == "__main__":
    main()
