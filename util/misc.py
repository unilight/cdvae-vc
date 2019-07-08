
# Based on 2017 Tomoki Hayashi (Nagoya University)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

import json
import os
import sys
from datetime import datetime

import fnmatch
import h5py
import numpy as np

import tensorflow as tf
import logging

def read_hdf5(hdf5_name, hdf5_path):
    """FUNCTION TO READ HDF5 DATASET

    Args:
        hdf5_name (str): filename of hdf5 file
        hdf5_path (str): dataset name in hdf5 file

    Return:
        dataset values
    """
    if not os.path.exists(hdf5_name):
        print("ERROR: There is no such a hdf5 file. (%s)" % hdf5_name)
        print("Please check the hdf5 file path.")
        sys.exit(-1)

    hdf5_file = h5py.File(hdf5_name, "r")

    if hdf5_path not in hdf5_file:
        print("ERROR: There is no such a data in hdf5 file. (%s)" % hdf5_path)
        print("Please check the data path in hdf5 file.")
        sys.exit(-1)

    hdf5_data = hdf5_file[hdf5_path].value
    hdf5_file.close()

    return hdf5_data


def write_hdf5(hdf5_name, hdf5_path, write_data, is_overwrite=True):
    """FUNCTION TO WRITE DATASET TO HDF5

    Args :
        hdf5_name (str): hdf5 dataset filename
        hdf5_path (str): dataset path in hdf5
        write_data (ndarray): data to write
        is_overwrite (bool): flag to decide whether to overwrite dataset
    """
    # convert to numpy array
    write_data = np.array(write_data)

    # check folder existence
    folder_name, _ = os.path.split(hdf5_name)
    if not os.path.exists(folder_name) and len(folder_name) != 0:
        os.makedirs(folder_name)

    # check hdf5 existence
    if os.path.exists(hdf5_name):
        # if already exists, open with r+ mode
        hdf5_file = h5py.File(hdf5_name, "r+")
        # check dataset existence
        if hdf5_path in hdf5_file:
            if is_overwrite:
                print("Warning: data in hdf5 file already exists. recreate dataset in hdf5.")
                hdf5_file.__delitem__(hdf5_path)
            else:
                print("ERROR: there is already dataset.")
                print("if you want to overwrite, please set is_overwrite = True.")
                hdf5_file.close()
                sys.exit(1)
    else:
        # if not exists, open with w mode
        hdf5_file = h5py.File(hdf5_name, "w")

    # write data to hdf5
    hdf5_file.create_dataset(hdf5_path, data=write_data)
    hdf5_file.flush()
    hdf5_file.close()


def find_files(directory, pattern="*.wav", use_dir_name=True):
    """FUNCTION TO FIND FILES RECURSIVELY

    Args:
        directory (str): root directory to find
        pattern (str): query to find
        use_dir_name (bool): if False, directory name is not included

    Return:
        (list): list of found filenames
    """
    files = []
    for root, dirnames, filenames in os.walk(directory, followlinks=True):
        for filename in fnmatch.filter(filenames, pattern):
            files.append(os.path.join(root, filename))
    if not use_dir_name:
        files = [file_.replace(directory + "/", "") for file_ in files]
    return files


def read_txt(file_list):
    """FUNCTION TO READ TXT FILE

    Arg:
        file_list (str): txt file filename

    Return:
        (list): list of read lines
    """
    with open(file_list, "r") as f:
        filenames = f.readlines()
    return [filename.replace("\n", "") for filename in filenames]

def save(saver, sess, logdir, step):
    ''' Save a model to logdir/model.ckpt-[step] '''
    model_name = 'model.ckpt'
    checkpoint_path = os.path.join(logdir, model_name)
    print('Storing checkpoint to {} ...'.format(logdir), end="")
    sys.stdout.flush()

    if not os.path.exists(logdir):
        os.makedirs(logdir)

    saver.save(sess, checkpoint_path, global_step=step)
    print(' Done.')


def load(saver, sess, logdir, ckpt=None):
    '''
    Try to load model form a dir (search for the newest checkpoint)
    '''
    if ckpt:
        ckpt = os.path.join(logdir, ckpt)
        global_step = int(ckpt.split('/')[-1].split('-')[-1])
        logging.info('  Global step: {}'.format(global_step))
        saver.restore(sess, ckpt)
        return global_step
    else:
        ckpt = tf.train.latest_checkpoint(logdir)
        if ckpt:
            logging.info('  Checkpoint found: {}'.format(ckpt))
            global_step = int(ckpt.split('/')[-1].split('-')[-1])
            logging.info('  Global step: {}'.format(global_step))
            saver.restore(sess, ckpt)
            return global_step
        else:
            print('No checkpoint found')
            return None

def get_default_logdir_train(note, logdir_root='logdir'):
    STARTED_DATESTRING = datetime.now().strftime('%0m%0d-%0H%0M-%0S-%Y')
    logdir = os.path.join(logdir_root, '{}-{}'.format(STARTED_DATESTRING, note))
    print('Using default logdir: {}'.format(logdir))        
    return logdir

def get_default_logdir_output(args):
    STARTED_DATESTRING = datetime.now().strftime('%0m%0d-%0H%0M-%0S-%Y')
    logdir = os.path.join(args.logdir, STARTED_DATESTRING+'-{}-{}'.format(args.src, args.trg))
    print('Logdir: {}'.format(logdir))        
    return logdir

class ValueWindow():
    def __init__(self, window_size=100):
        self._window_size = window_size
        self._values = []

    def append(self, x):
        self._values = self._values[-(self._window_size - 1):] + [x]

    @property
    def sum(self):
        return sum(self._values)

    @property
    def count(self):
        return len(self._values)

    @property
    def average(self):
        return self.sum / max(1, self.count)

    def reset(self):
        self._values = []
