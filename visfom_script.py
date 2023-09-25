#!/usr/bin/env python3
import argparse
import logging

import numpy as np
import matplotlib.pyplot as plt

import vis_access as vis


def initialize_logs():
    """
    Initialize the log settings
    """
    logging.basicConfig(format='%(message)s', level=logging.INFO)


def create_parser():
    parser = argparse.ArgumentParser(description="This package enhance the quality and accuracy of MeerKAT's"
                                     " observations by automatically identifying and mitigating radio-frequency"
                                     " interference from complex-valued visibility data.")
    parser.add_argument('-p', '--path', action='store', type=str,
                       help='A path to the rdb file')
    return parser


def main():
    # Initializing the log settings
    initialize_logs()
    logging.info('Visibility Foundation Model Framework')
    parser = create_parser()
    args = parser.parse_args()
    path = args.path
    logging.info('Read in the rdb file')
    data = vis.read_rdb(path)
    for scan in data.scans():
        n_time, n_chans, n_bl = data.shape
        vis_chunk = np.empty((int(n_time), int(n_chans), int(n_bl)), dtype=np.complex64)
        weight_chunk = np.zeros_like(vis_chunk, dtype=np.float32)
        flag_chunk = np.zeros_like(vis_chunk, dtype=np.bool)
        vis.load(data, np.s_[:, :, :], vis_chunk, weight_chunk, flag_chunk)
        plt.imshow(np.mean(np.abs(vis_chunk),axis=2), aspect='auto')
        plt.show(block=True)


if __name__=="__main__":
    main()