#!/usr/bin/python3
# -*- coding: utf-8 -*-
#
# Developed by Haozhe Xie <cshzxie@gmail.com>

import logging
import matplotlib
import multiprocessing as mp
import numpy as np
import os
import sys
from argparse import ArgumentParser
from pprint import pprint
from config import cfg
from core.attack_sgd import test_net

# Fix problem: no $DISPLAY environment variable
matplotlib.use('Agg')

# all_targets = ['chair', 'desk', 'car', 'tab', 'case', 'mail']
# all_sources = ['chair', 'desk', 'car', 'tab', 'case', 'mail']

# all_targets = ['desk']
# all_sources = ['chair']
all_targets = ['chair']
all_sources = ['tab']
# all_targets = ['tab']
# all_sources = ['chair']
def get_args_from_command_line():
    parser = ArgumentParser(description='Parser of Runner of Pix2Vox')
    parser.add_argument('--gpu',
                        dest='gpu_id',
                        help='GPU device id to use [cuda0]',
                        default=cfg.CONST.DEVICE,
                        type=str)
    parser.add_argument('--rand', dest='randomize', help='Randomize (do not use a fixed seed)', action='store_true')
    parser.add_argument('--weights', dest='weights', help='Initialize network from the weights file',
                        default='Pix2Vox-A-ShapeNet.pth')
    parser.add_argument('--out', dest='out_path', help='Set output path', default=cfg.DIR.OUT_PATH)
    parser.add_argument('--weight', dest='weight', help='Set output path', default=10, type=int)
    parser.add_argument('--source', type=str, nargs='+')
    parser.add_argument('--target', type=str, nargs='+')

    parser.add_argument("--alpha_inp", default=0.0, type=float, help="step size for input")
    parser.add_argument("--alpha_flow", type=float, default=0.0, help="step size for flow")
    parser.add_argument("--tau", type=float, default=0.0, help="tau for flow")
    parser.add_argument("--attack_epsilon", type=float, help="Linf ball for perturbation")  # [0-255]
    parser.add_argument("--attack_type", type=str, help="type of attack", choices=['spatial_dag', 'dag', 'spatial'])
    parser.add_argument("--foreground_only", type=int, help="foreground/background attack", default=0)
    args = parser.parse_args()
    return args


def main():
    # Get args from command line
    args = get_args_from_command_line()

    if args.gpu_id is not None:
        cfg.CONST.DEVICE = args.gpu_id
    if not args.randomize:
        np.random.seed(cfg.CONST.RNG_SEED)
    if args.out_path is not None:
        cfg.DIR.OUT_PATH = args.out_path
    if args.weights is not None:
        cfg.CONST.WEIGHTS = args.weights

        # Print config
    print('Use config:')
    pprint(cfg)

    # Set GPU to use
    if type(cfg.CONST.DEVICE) == str:
        os.environ["CUDA_VISIBLE_DEVICES"] = cfg.CONST.DEVICE
    
    all_sources = args.source
    all_targets = args.target
    source_list = []
    target_list = []
    for source in all_sources:
        for target in all_targets:
            if source != target:
                source_list.append(source)
                target_list.append(target)
    test_net(cfg, sources=source_list, targets=target_list, weight=args.weight, args=args)


if __name__ == '__main__':
    # Check python version
    if sys.version_info < (3, 0):
        raise Exception("Please follow the installation instruction on 'https://github.com/hzxie/Pix2Vox'")

    # Setup logger
    mp.log_to_stderr()
    logger = mp.get_logger()
    logger.setLevel(logging.INFO)

    main()