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
# Fix problem: no $DISPLAY environment variable
matplotlib.use('Agg')

from argparse import ArgumentParser
from datetime import datetime as dt
from pprint import pprint

from config import cfg
from core.train import train_net
#from core.test import test_net
#from core.test_single_image import test_net
from core.attack import test_net
from core.forward_pass import forward_pass
from core.forward_pass_npy import forward_pass_npy

# all_targets = ['chair', 'desk', 'car', 'tab', 'case', 'mail']
# all_sources = ['chair', 'desk', 'car', 'tab', 'case', 'mail']

all_targets = ['chair']
all_sources = ['tab']

def get_args_from_command_line():
    parser = ArgumentParser(description='Parser of Runner of Pix2Vox')
    parser.add_argument('--gpu',
                        dest='gpu_id',
                        help='GPU device id to use [cuda0]',
                        default=cfg.CONST.DEVICE,
                        type=str)
    parser.add_argument('--rand', dest='randomize', help='Randomize (do not use a fixed seed)', action='store_true')
    parser.add_argument('--test', dest='test', help='Test neural networks', action='store_true')
    parser.add_argument('--batch-size',
                        dest='batch_size',
                        help='name of the net',
                        default=cfg.CONST.BATCH_SIZE,
                        type=int)
    parser.add_argument('--epoch', dest='epoch', help='number of epoches', default=cfg.TRAIN.NUM_EPOCHES, type=int)
    parser.add_argument('--weights', dest='weights', help='Initialize network from the weights file', default='Pix2Vox-A-ShapeNet.pth')
    parser.add_argument('--out', dest='out_path', help='Set output path', default=cfg.DIR.OUT_PATH)
    parser.add_argument('--source', dest='source', help='Set output path', default='chair')
    parser.add_argument('--target', dest='target', help='Set output path', default='ones')
    parser.add_argument('--img_path', dest='img_path', help='Set output path', default='chair_ones/')
    parser.add_argument('--weight', dest='weight', help='Set output path', default=10, type=int)
    parser.add_argument('--forward_pass', dest='forward_pass', help='Set output path', default=0, type=int)
    parser.add_argument('--forward_pass_npy', dest='forward_pass_npy', help='Set output path', default=0, type=int)
    parser.add_argument('--x_path', dest='x_path', help='Set output path', default="")
    parser.add_argument('--f_path', dest='f_path', help='Set output path', default="")

    parser.add_argument("--alpha_inp", default=0.0, type=float, help="step size for input")
    parser.add_argument("--alpha_flow", type=float, default=0.0, help="step size for flow")
    parser.add_argument("--tau", type=float, default=0.0, help="tau for flow")
    parser.add_argument("--attack_epsilon", type=int, help="Linf ball for perturbation") #[0-255]
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
    if args.batch_size is not None:
        cfg.CONST.BATCH_SIZE = args.batch_size
    if args.epoch is not None:
        cfg.TRAIN.NUM_EPOCHES = args.epoch
    if args.out_path is not None:
        cfg.DIR.OUT_PATH = args.out_path
    if args.weights is not None:
        cfg.CONST.WEIGHTS = args.weights
        if not args.test:
            cfg.TRAIN.RESUME_TRAIN = True

    # Print config
    print('Use config:')
    pprint(cfg)

    # Set GPU to use
    if type(cfg.CONST.DEVICE) == str:
        os.environ["CUDA_VISIBLE_DEVICES"] = cfg.CONST.DEVICE

    # Start train/test process
    if not args.test:
        train_net(cfg)
    else:
        #test_net(cfg)
        source_list = []
        target_list = []
        for source in all_sources:
            for target in all_targets:
                if source != target:
                    source_list.append(source)
                    target_list.append(target)
        test_net(cfg, sources=source_list, targets=target_list, weight=args.weight, args=args)
        # if 'WEIGHTS' in cfg.CONST and os.path.exists(cfg.CONST.WEIGHTS):
        #     if args.forward_pass_npy==1:
        #         forward_pass_npy(cfg, x_path=args.x_path, f_path=args.f_path)
        #         sys.exit(0)
        #     if args.forward_pass==1:
        #         forward(cfg, source=args.source, img_path=args.img_path)
        #     else:
        #         if args.target == 'all':
        #             for target in all_targets:
        #                 if target == args.source:
        #                     continue
        #                 test_net(cfg, source=args.source, target=target, weight=args.weight)
        #         else:
        #             test_net(cfg, source=args.source, target=args.target, weight=args.weight)
        # else:
        #     print('[FATAL] %s Please specify the file path of checkpoint.' % (dt.now()))
        #     sys.exit(2)


if __name__ == '__main__':
    # Check python version
    if sys.version_info < (3, 0):
        raise Exception("Please follow the installation instruction on 'https://github.com/hzxie/Pix2Vox'")

    # Setup logger
    mp.log_to_stderr()
    logger = mp.get_logger()
    logger.setLevel(logging.INFO)

    main()
