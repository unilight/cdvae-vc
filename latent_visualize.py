import os 
import json

import numpy as np 
import tensorflow as tf 

from matplotlib import pyplot as plt
plt.switch_backend('agg')

import argparse
import logging
import multiprocessing as mp

def visualize(file_list, arch, output_dir, ):
    
    for bin_path in file_list:

        basename = os.path.splitext(os.path.split(bin_path)[-1])[0]
        latent = np.fromfile(bin_path, dtype=np.float32).reshape([-1, arch['z_dim']])

        fig = plt.figure()
        cm = plt.get_cmap('Oranges')
        px = np.arange(latent.shape[0])
        py = np.arange(latent.shape[1]+1)
        plt.pcolor(px, py, latent.T, cmap=cm)
        plt.colorbar()
        plt.savefig(os.path.join(output_dir, basename+'.jpg'))
        plt.close()

        logging.info(basename)

def main():
    parser = argparse.ArgumentParser(
        description="visualize latent code")    
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
    logging.info('Latent visualization')
    logging.info(args)

    train_dir = os.sep.join(args.logdir.split(os.sep)[:-2])
    output_dir = os.path.join(args.logdir, 'latent-visualize')

    # Load architecture
    arch = tf.gfile.Glob(os.path.join(train_dir, 'architecture*.json'))[0]  # should only be 1 file
    with open(arch) as fp:
        arch = json.load(fp)

    # make directories for output
    tf.gfile.MakeDirs(output_dir)

    # Get and divide list
    bin_list = sorted(tf.gfile.Glob(os.path.join(args.logdir, 'latent', '*.bin')))
    logging.info("number of utterances = %d" % len(bin_list))
    file_lists = np.array_split(bin_list, args.n_jobs)
    file_lists = [f_list.tolist() for f_list in file_lists]

    # multi processing
    with mp.Manager() as manager:
        processes = []
        for f in file_lists:
            p = mp.Process(target=visualize, args=(f, arch, output_dir, ))
            p.start()
            processes.append(p)

        # wait for all process
        for p in processes:
            p.join()

if __name__ == '__main__':
    main()