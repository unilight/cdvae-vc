#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Feature extraction (WORLD features) for VCC2018 dataset
# By Wen-Chin Huang 2019.06
# Based on 2017 Tomoki Hayashi (Nagoya University)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

import argparse
import logging
import multiprocessing as mp
import os
import sys
import copy

from distutils.util import strtobool

import numpy as np
import pysptk
import pyworld as pw
import librosa

from numpy.matlib import repmat
from scipy.interpolate import interp1d
from scipy.io import wavfile
from scipy.signal import firwin
from scipy.signal import lfilter
from sprocket.speech.feature_extractor import FeatureExtractor

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from util.misc import (find_files, read_txt, read_hdf5, write_hdf5)

def energy_norm(feat):
    en = np.sum(feat + 1e-8, axis=1, keepdims=True)
    feat = feat / en
    return en, feat

def low_cut_filter(x, fs, cutoff=70):
    """FUNCTION TO APPLY LOW CUT FILTER

    Args:
        x (ndarray): Waveform sequence
        fs (int): Sampling frequency
        cutoff (float): Cutoff frequency of low cut filter

    Return:
        (ndarray): Low cut filtered waveform sequence
    """

    nyquist = fs // 2
    norm_cutoff = cutoff / nyquist

    # low cut filter
    fil = firwin(255, norm_cutoff, pass_zero=False)
    lcf_x = lfilter(fil, 1, x)

    return lcf_x

def filepath_create(wav_list, bindir):
    """CREATE FILE FOLDER"""
    for wav_name in wav_list:
        spk = os.path.dirname(wav_name).split('/')[-1][-3:]
        novad_bin_dir = os.path.join(bindir, 'noVAD', spk)
        vad_bin_dir = os.path.join(bindir, 'VAD', spk)
        
        if not os.path.exists(novad_bin_dir):
            os.makedirs(novad_bin_dir)
        if not os.path.exists(vad_bin_dir):
            os.makedirs(vad_bin_dir)

def world_feature_extract(wav_list, spk_list, feat_param_list, args):
    """EXTRACT WORLD FEATURE VECTOR"""

    for i, wav_name in enumerate(wav_list):
        bin_basename = os.path.basename(wav_name).replace('wav', 'bin')
        spk = os.path.dirname(wav_name).split('/')[-1][-3:]
        bin_name = os.path.join(args.bindir, 'noVAD', spk, bin_basename)
        vad_bin_name = os.path.join(args.bindir, 'VAD', spk, bin_basename)

        if os.path.exists(bin_name):
            if args.overwrite:
                logging.info("overwrite %s (%d/%d)" % (wav_name, i + 1, len(wav_list)))
            else:
                logging.info("skip %s (%d/%d)" % (wav_name, i + 1, len(wav_list)))            
                continue
        else:
            logging.info("now processing %s (%d/%d)" % (wav_name, i + 1, len(wav_list)))
        
        feat_param = feat_param_list[spk_list.index(spk)]
        
        # load wavfile and apply low cut filter
        fs, x = wavfile.read(wav_name)
        x = np.array(x, dtype=np.float64)
        x = low_cut_filter(x, fs, cutoff=feat_param['highpass_cutoff'])

        # check sampling frequency
        if not fs == feat_param['fs']:
            logging.error("sampling frequency is not matched.")
            sys.exit(1)

        # extract features
        f0, time_axis = pw.harvest(x, feat_param['fs'],
                                   f0_floor=feat_param['f0min'],
                                   f0_ceil=feat_param['f0max'],
                                   frame_period=feat_param['shift_ms'])
        sp = pw.cheaptrick(x, f0, time_axis, feat_param['fs'],
                           fft_size=feat_param['fftl'])
        ap = pw.d4c(x, f0, time_axis, feat_param['fs'], fft_size=feat_param['fftl'])
        mcc = pysptk.sp2mc(sp, feat_param['mcep_dim'], feat_param['mcep_alpha'])
        en_sp, sp = energy_norm(sp)
        sp = np.log10(sp)
        en_mcc = mcc[:, 0]

        # expand dimensions for concatenation
        f0 = np.expand_dims(f0, axis=-1)
        en_mcc = np.expand_dims(en_mcc, axis=-1)

        # concatenation
        world_feats = np.concatenate([sp, mcc[:, 1:], ap, f0, en_sp, en_mcc], axis=1)
        labels = spk_list.index(spk) * np.ones(
                [sp.shape[0], 1], np.float32)
        
        # concatenate all features
        feats = np.concatenate(
                    [world_feats, labels],
                    axis=1).astype(np.float32)

        # VAD
        vad_idx = np.where(f0.copy().reshape([-1])>10)[0]
        vad_feats = feats[vad_idx[0] : vad_idx[-1]+1]

        # write to bin
        with open(bin_name, 'wb') as fp:
            fp.write(feats.tostring())
        with open(vad_bin_name, 'wb') as fp:
            fp.write(vad_feats.tostring())


def main():
    parser = argparse.ArgumentParser(
        description="making feature file argsurations.")

    parser.add_argument(
        "--waveforms", required=True, type=str,
        help="directory or list of filename of input wavfile")
    parser.add_argument(
        "--bindir", required=True, type=str,
        help="directory to save bin")
    parser.add_argument(
        "--confdir", required=True, type=str,
        help="configuration directory")
    parser.add_argument(
        "--overwrite", default=False,
        type=strtobool, help="if set true, overwrite the exist feature files")
    parser.add_argument(
        "--n_jobs", default=12,
        type=int, help="number of parallel jobs")
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

    # read list
    if os.path.isdir(args.waveforms):
        file_list = sorted(find_files(args.waveforms, "*.wav"))
    else:
        file_list = read_txt(args.waveforms)
    logging.info("number of utterances = %d" % len(file_list))

    # read speaker list
    spk_list = read_txt(os.path.join(args.confdir, 'spk.list'))

    # read f0 max/min of the speaker, and define feature extractor
    feat_param_list = []
    for s in spk_list:
        with open(args.confdir + '/' + s + '.f0', 'r') as f:
            f0min, f0max = [int(f0) for f0 in f.read().split(' ')]
        feat_param_list.append({
                'fs'                : 22050,
                'shift_ms'          : 5,
                'length_ms'         : 25,
                'fftl'              : 1024,
                'n_mels'            : 80,
                'mcep_dim'          : 34,
                'mcep_alpha'        : 0.455,
                'f0min'             : f0min,
                'f0max'             : f0max,
                'highpass_cutoff'   : 70,
        })

    # create file folders
    filepath_create(file_list, args.bindir)
    # divide list
    file_lists = np.array_split(file_list, args.n_jobs)
    file_lists = [f_list.tolist() for f_list in file_lists]

    # multi processing
    processes = []
    target_fn = world_feature_extract
    for f in file_lists:
        p = mp.Process(target=target_fn, args=(f, spk_list, feat_param_list, args,))
        p.start()
        processes.append(p)

    # wait for all process
    for p in processes:
        p.join()


if __name__ == "__main__":
    main()
