'''
Demo code for the paper

Choy et al., 3D-R2N2: A Unified Approach for Single and Multi-view 3D Object
Reconstruction, ECCV 2016
'''

import os
import sys
import theano
if (sys.version_info < (3, 0)):
    raise Exception("Please follow the installation instruction on 'https://github.com/chrischoy/3D-R2N2'")

import shutil
import numpy as np
from subprocess import call
    
from PIL import Image
from models import load_model
from lib.config import cfg, cfg_from_list
from lib.solver import Solver
from lib.voxel import voxel2obj
import argparse

DEFAULT_WEIGHTS = 'output/ResidualGRUNet/default_model/weights.npy'

def cmd_exists(cmd):
    return shutil.which(cmd) is not None

def download_model(fn):
    if not os.path.isfile(fn):
        # Download the file if doewn't exist
        print('Downloading a pretrained model')
        call(['curl', 'ftp://cs.stanford.edu/cs/cvgl/ResidualGRUNet.npy',
              '--create-dirs', '-o', fn])

def load_images(obj):
    ims = []
    for i in range(3):
        im = Image.open('imgs/' + obj + str(i) + '.png')
        ims.append([np.array(im).transpose(
            (2, 0, 1)).astype(np.float32) / 255.])
    return np.array(ims)

def get_foreground(imgs):
    foreground_mask = (imgs<1)
    return foreground_mask
    #foreground_imgs = np.zeros()


def is_boundary(pos, shape):
    return pos[0] == 0 or pos[1] == 0 or pos[2] == 0 or pos[0] == shape[0]-1 or pos[1] == shape[1]-1 \
           or pos[2] == shape[2] -1

def main(args):
    '''Main demo function'''
    # Save prediction into a file named 'prediction.obj' or the given argument
    #pred_file_name = sys.argv[1] if len(sys.argv) > 1 else 'prediction.obj'

    download_model(DEFAULT_WEIGHTS)
    # Use the default network model
    NetClass = load_model('ResidualGRUNet')
    # Define a network and a solver. Solver provides a wrapper for the test function.
    net = NetClass(compute_grad=False)  # instantiate a network

    net.load(DEFAULT_WEIGHTS) # load downloaded weights

    targets = args.target
   
    solver = Solver(net) # instantiate a solver

    source_type = args.source

    background_attack = (args.background != 0)

    # load images
    demo_imgs = load_images(source_type)

    x_adv = demo_imgs
    flow = np.zeros((demo_imgs.shape[0], cfg.CONST.BATCH_SIZE, 2, demo_imgs.shape[-2], demo_imgs.shape[-1])).astype(theano.config.floatX)

    target_imgs = load_images(args.target)
    target,_ = solver.test_output(target_imgs, flow=flow)
    #target = np.zeros((cfg.CONST.BATCH_SIZE, 32, 2, 32, 32)).astype(theano.config.floatX)
    #target[:, :, 1, :, :] = 1
    if not os.path.exists('targets'):
        os.makedirs('targets')
    voxel2obj('targets/' + args.target + '.obj', target[0, :, 1, :, :] > cfg.TEST.VOXEL_THRESH)

    voxel_prediction, _ = solver.spatial_dag_attack(x_adv, flow, args.save_dir, target=target, max_iters=1000, attack_epsilon=args.attack_epsilon/255, tau=args.tau, 
        alpha_flow=args.alpha_flow, alpha_inp=args.alpha_inp, source_img=source_type, target_img=args.target, 
        background_attack=background_attack, weight=args.border_weight, 
        dag_only=args.dag_only, spatial_only=args.spatial_only)

    print('savng prediction')

if __name__ == '__main__':
    # Set the batch size to 1
    cfg_from_list(['CONST.BATCH_SIZE', 1])
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=str, default='chair', help='source image')
    parser.add_argument('--target', type=str, default='all', help='target image')
    parser.add_argument('--background', type=int, default=1, help='attack background')    
    parser.add_argument('--border_weight', type=int, default=2, help='border voxels weight')
    parser.add_argument('--alpha_flow', type=float, default=5, help='flow alpha')
    parser.add_argument('--alpha_inp', type=float, default=0.0005, help='input alpha')
    parser.add_argument('--tv_weight', type=float, default=0.0, help='tv loss weight')
    parser.add_argument('--attack_epsilon', type=int, default=1, help='pgd max perturbation')
    parser.add_argument('--tau', type=float, default=0.0, help='flow regularization term')
    parser.add_argument('--dag_only', type=int, default=0, help='ablation_study for dag')
    parser.add_argument('--spatial_only', type=int, default=0, help='ablation_study for spatial')
    parser.add_argument('--save_dir', type=str, default='', help='directory to save results')

    args = parser.parse_args()
    cfg_from_list(['CONST.TV_LOSS_WEIGHT', args.tv_weight])
    main(args)
