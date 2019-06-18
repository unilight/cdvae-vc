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
