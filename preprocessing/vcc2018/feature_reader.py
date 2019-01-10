#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import division

import argparse
import os
import sys
import numpy as np
import h5py
import logging
from scipy.io import wavfile
from sprocket.speech.synthesizer import Synthesizer

import tensorflow as tf

def world_synthesis(wav_name, feat_param, f0, ap, spectral, spectral_type):
    """WORLD SPEECH SYNTHESIS
    Args:
        wav_name (str): filename of synthesised wav
        feat_param (dict): acoustic feature parameter dictionary
        f0(np array): pitch features
        ap: aperiodicity features
        spectral: spectral features
        spectral_type: spectral feature type (sp or mcc)
    """
    synthesizer = Synthesizer(fs=feat_param['fs'],
                              fftl=feat_param['fftl'],
                              shiftms=feat_param['shiftms'])

    if spectral_type == 'mcc':
        wav = synthesizer.synthesis(f0,
                                    spectral,
                                    ap,
                                    alpha=feat_param['mcep_alpha'])
    elif spectral_type == 'sp':
        wav = synthesizer.synthesis_spc(f0,
                                    spectral,
                                    ap)
    else:
        logging.info("Currently support 'mcep' or 'spc' only.")
        raise ValueError

    wav = np.clip(wav, -32768, 32767)
    wavfile.write(wav_name, feat_param['fs'], wav.astype(np.int16))
    logging.info("wrote %s." % (wav_name))

def Frame_feature_reader(
    file_pattern,
    feat_param,
    batch_size,
    capacity=256,
    min_after_dequeue=128,
    num_threads=8,
    ):
    
    with tf.name_scope('InputSpectralFrame'):
       
        # get dimensions
        SP_DIM = feat_param['sp_dim']
        MCC_DIM = feat_param['mcc_dim']
        FEAT_DIM = feat_param['feat_dim']
        record_bytes = FEAT_DIM * 4
        
        files = []
        for p in file_pattern:
            files.extend(tf.gfile.Glob(p))

        logging.info('Found {} files'.format(len(files)))
        print('Found {} files'.format(len(files)))
        
        filename_queue = tf.train.string_input_producer(files)

        reader = tf.FixedLengthRecordReader(record_bytes)
        _, value = reader.read(filename_queue)
        value = tf.decode_raw(value, tf.float32)

        value = tf.reshape(value, [FEAT_DIM,])
        sp       = value[:SP_DIM]
        mcc      = value[SP_DIM:SP_DIM + MCC_DIM]
        f0       = value[SP_DIM * 2 + MCC_DIM * 3]
        uv       = value[SP_DIM * 2 + MCC_DIM * 3 + 2]
        cont_f0  = value[SP_DIM * 2 + MCC_DIM * 3 + 3]
        bandap   = value[-3:-1]
        speaker  = tf.cast(value[-1], tf.int64)

        dictionary = {
            'sp': sp, 
            'mcc': mcc,
            'f0': f0, 
            'uv': uv, 
            'cont_f0': cont_f0,
            'bandap': bandap, 
            'speaker': speaker,
        }
        
        return tf.train.shuffle_batch(
            dictionary,
            batch_size,
            capacity=capacity,
            min_after_dequeue=min_after_dequeue,
            num_threads=num_threads,
        )

def Whole_feature_reader(filename, feat_param, dtype=np.float32):
    """FUNCTION TO READ whole utterance of features
    """
    SP_DIM = feat_param['sp_dim']
    MCC_DIM = feat_param['mcc_dim']
    FEAT_DIM = feat_param['feat_dim']

    values = np.fromfile(filename, dtype).astype(np.float64).reshape([-1, FEAT_DIM])

    sp       = values[:,                  : SP_DIM].copy(order='C')
    mcc      = values[:, SP_DIM           : SP_DIM + MCC_DIM].copy(order='C')
    ap       = values[:, SP_DIM + MCC_DIM * 3 : SP_DIM * 2 + MCC_DIM * 3].copy(order='C')
    f0       = values[:, SP_DIM * 2 + MCC_DIM * 3].copy(order='C')
    en_sp    = values[:, SP_DIM * 2 + MCC_DIM * 3 + 1].copy(order='C')
    en_mcc   = values[:, SP_DIM * 2 + MCC_DIM * 3 + 2].copy(order='C')
    acoustic = values[:, SP_DIM * 2 + MCC_DIM * 3 + 2 : -1].copy(order='C')
    speaker  = values[:, -1].astype(np.int64)

    dictionary = {
        'sp': sp, 
        'mcc': mcc,
        'ap': ap,
        'f0': f0, 
        'en_sp': en_sp,
        'en_mcc': en_mcc, 
        'acoustic': acoustic,
        'speaker': speaker,
    }

    return dictionary

def Whole_feature_reader_tf(file_pattern, feat_param, num_epochs=1):
    """FUNCTION TO DEFINE THE READER using tensorflow built-in API
    """
    SP_DIM = feat_param['sp_dim']
    MCC_DIM = feat_param['mcc_dim']
    FEAT_DIM = feat_param['feat_dim']
    
    files = tf.gfile.Glob(file_pattern)
    print('Found {} files'.format(len(files)))
    filename_queue = tf.train.string_input_producer(files, num_epochs=num_epochs)
    reader = tf.WholeFileReader()
    key, value = reader.read(filename_queue)
    value = tf.decode_raw(value, tf.float32)
    values = tf.reshape(value, [-1, FEAT_DIM])
    sp      = values[:,                  : SP_DIM]
    mcc     = values[:, SP_DIM           : SP_DIM + MCC_DIM]
    ap      = values[:, SP_DIM + MCC_DIM : SP_DIM * 2 + MCC_DIM]
    f0      = values[:, SP_DIM * 2 + MCC_DIM]
    en_sp   = values[:, SP_DIM * 2 + MCC_DIM + 1]
    en_mcc  = values[:, SP_DIM * 2 + MCC_DIM + 2]
    speaker = tf.cast(values[:, -1], tf.int64)
    filename = key
        
    dictionary = {
        'sp': sp, 
        'mcc': mcc,
        'ap': ap,
        'f0': f0, 
        'en_sp': en_sp,
        'en_mcc': en_mcc, 
        'speaker': speaker,
        'filename': filename,
        'num_files': len(files),
    }

    return dictionary


def main():
    """ Feature reader & synthesis check
    Usage: 
         1. read original features
         feature_ready.py --filename filename 
         2. read f0 transformed features
         feature_ready.py --filename filename --tarspk target_speaker
    """
    parser = argparse.ArgumentParser(
        description="test feature readers")
    parser.add_argument(
        "--file_pattern", default=None, type=str,
        help="the pattern of the testing feature file(s)")
    parser.add_argument(
        "--tarspk", default=None, type=str,
        help="the name of target speaker")
    parser.add_argument(
        "--wavname", default='test.wav', type=str,
        help="the name of output wav")
    args = parser.parse_args()

    # parameter setting
    feat_param = {
        'fs':22050,
        'shiftms':5,
        'fftl':1024,
        'mcep_alpha': 0.455,
        'sp_dim':513,
        'mcc_dim':34,
        'feat_dim': 513 + 34 + 513 + 3 + 39 + 1
    }
    # load acoustic features and synthesis 
    if os.path.exists(args.file_pattern):
        sp, mcc, ap, f0, en_sp, en_mcc, acoustic, spk, = Whole_feature_reader(
                args.file_pattern, feat_param)
        en_mcc = np.expand_dims(en_mcc, 1)
        mcc = np.concatenate([en_mcc, mcc], axis=1)
        world_synthesis(args.wavname, feat_param, f0, mcc, ap)

if __name__ == "__main__":
    main()
