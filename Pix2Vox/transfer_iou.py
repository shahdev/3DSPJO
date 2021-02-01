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
from core.test_single_image import test_net
from core.forward_pass import forward_pass

all_targets = ['chair', 'desk', 'car', 'tab', 'case', 'mail', 'ones']
#all_sources = ['chair', 'desk', 'car', 'tab', 'case', 'mail', 'ones']
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
    parser.add_argument('--weights', dest='weights', help='Initialize network from the weights file', default='/home/hari/Pix2Vox-A-ShapeNet.pth')
    parser.add_argument('--out', dest='out_path', help='Set output path', default=cfg.DIR.OUT_PATH)
    parser.add_argument('--source', dest='source', help='Set output path', default='chair')
    parser.add_argument('--target', dest='target', help='Set output path', default='ones')
    parser.add_argument('--img_path', dest='img_path', help='Set output path', default='chair_ones/')
    parser.add_argument('--weight', dest='weight', help='Set output path', default=10, type=int)
    parser.add_argument('--forward_pass', dest='forward_pass', help='Set output path', default=0, type=int)
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

    # Set GPU to use
    if type(cfg.CONST.DEVICE) == str:
        os.environ["CUDA_VISIBLE_DEVICES"] = cfg.CONST.DEVICE

    # Start train/test process
    if not args.test:
        train_net(cfg)
    else:
        if 'WEIGHTS' in cfg.CONST and os.path.exists(cfg.CONST.WEIGHTS):
            for src in all_sources:
                for targ in all_targets:
                    if  src== targ:
                        continue
                    try:
                        iou = forward_pass(cfg, source=src, img_path='../pix2voxf/%s_%s/pixel_attack_399'
                                                                     '99_'%(src, targ))
                        print(src, targ, iou)
                    except:
                        print("e")
                        continue
        else:
            print('[FATAL] %s Please specify the file path of checkpoint.' % (dt.now()))
            sys.exit(2)


if __name__ == '__main__':
    # Check python version
    if sys.version_info < (3, 0):
        raise Exception("Please follow the installation instruction on 'https://github.com/hzxie/Pix2Vox'")

    # Setup logger
    mp.log_to_stderr()
    logger = mp.get_logger()
    logger.setLevel(logging.INFO)

    main()
