# -*- coding: utf-8 -*-
#
# Developed by Haozhe Xie <cshzxie@gmail.com>

import json
import numpy as np
import os
import torch
import torch.backends.cudnn
import torch.utils.data
import sys
import utils.binvox_visualization
import utils.data_loaders
import utils.data_transforms
import utils.network_utils
import pdb
from datetime import datetime as dt
from PIL import Image
from models.encoder import Encoder
from models.decoder import Decoder
from models.refiner import Refiner
from models.merger import Merger

def load_images(img_path):
    ims = []
    for i in range(3):
        im = Image.open(img_path + str(i) + '.png')
        ims.append([np.array(im)[:,:,0:3].transpose(
            (2, 0, 1)).astype(np.float32) / 255.])
    return np.array(ims)

#record iou with source for untargeted attack
def forward_pass(cfg,
             epoch_idx=-1,
             output_dir=None,
             test_data_loader=None,
             test_writer=None,
             source=None,
             encoder=None,
             decoder=None,
             refiner=None,
             merger=None,
            img_path=None):
    # Enable the inbuilt cudnn auto-tuner to find the best algorithm to use
    torch.backends.cudnn.benchmark = True

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)
    demo_imgs = load_images(img_path)

    demo_imgs -= cfg.DATASET.MEAN[0]
    demo_imgs /= cfg.DATASET.STD[0]
    demo_imgs = np.transpose(demo_imgs, (1,0,2,3,4))
    demo_imgs = torch.tensor(demo_imgs, device=device).float()


    #ground_truth_volume = np.load('/home/hari/ground_truth/ground_truth_%s.npy'%source)
    source_imgs = load_images('/home/devansh/imgs/'+source)
    source_imgs -= cfg.DATASET.MEAN[0]
    source_imgs /= cfg.DATASET.STD[0]
    source_imgs = np.transpose(source_imgs, (1, 0, 2, 3, 4))
    source_imgs = torch.tensor(source_imgs, device=device).float()

    # Set up networks
    if decoder is None or encoder is None:
        encoder = Encoder(cfg)
        decoder = Decoder(cfg)
        refiner = Refiner(cfg)
        merger = Merger(cfg)

        if torch.cuda.is_available():
            encoder = torch.nn.DataParallel(encoder).cuda()
            decoder = torch.nn.DataParallel(decoder).cuda()
            refiner = torch.nn.DataParallel(refiner).cuda()
            merger = torch.nn.DataParallel(merger).cuda()

        print('[INFO] %s Loading weights from %s ...' % (dt.now(), cfg.CONST.WEIGHTS))
        checkpoint = torch.load(cfg.CONST.WEIGHTS)
        epoch_idx = checkpoint['epoch_idx']
        encoder.load_state_dict(checkpoint['encoder_state_dict'])
        decoder.load_state_dict(checkpoint['decoder_state_dict'])

        if cfg.NETWORK.USE_REFINER:
            refiner.load_state_dict(checkpoint['refiner_state_dict'])
        if cfg.NETWORK.USE_MERGER:
            merger.load_state_dict(checkpoint['merger_state_dict'])

    # Switch models to evaluation mode
    encoder.eval()
    decoder.eval()
    refiner.eval()
    merger.eval()

    image_features = encoder(source_imgs)
    raw_features, source_volume = decoder(image_features)

    if cfg.NETWORK.USE_MERGER and epoch_idx >= cfg.TRAIN.EPOCH_START_USE_MERGER:
        source_volume = merger(raw_features, source_volume)
    else:
        source_volume = torch.mean(source_volume, dim=1)

    if cfg.NETWORK.USE_REFINER and epoch_idx >= cfg.TRAIN.EPOCH_START_USE_REFINER:
        source_volume = refiner(source_volume)

    # Test the encoder, decoder, refiner and merger
    #image_features = encoder(rendering_images)
    image_features = encoder(demo_imgs)
    raw_features, generated_volume = decoder(image_features)

    if cfg.NETWORK.USE_MERGER and epoch_idx >= cfg.TRAIN.EPOCH_START_USE_MERGER:
        generated_volume = merger(raw_features, generated_volume)
    else:
        generated_volume = torch.mean(generated_volume, dim=1)

    if cfg.NETWORK.USE_REFINER and epoch_idx >= cfg.TRAIN.EPOCH_START_USE_REFINER:
        generated_volume = refiner(generated_volume)

    threshold = 0.4

    _volume = torch.ge(generated_volume, threshold)
    np_obj = _volume.data.cpu().data.numpy()
    #print('prediction sum: ', np.sum(np_obj))

    _volume_s = torch.ge(source_volume, threshold)
    np_obj_s = _volume_s.data.cpu().data.numpy()
    #print('prediction sum source: ', np.sum(np_obj_s))
    iou = (np_obj & np_obj_s).sum() / (np_obj | np_obj_s).sum()
    #print('iou: ', iou )
    voxel2obj('prediction.obj', np_obj[0])

    return iou

def evaluate_voxel_prediction(preds, gt, thresh):
    preds_occupy = preds[:, 1, :, :] >= thresh
    diff = np.sum(np.logical_xor(preds_occupy, gt[:, 1, :, :]))
    intersection = np.sum(np.logical_and(preds_occupy, gt[:, 1, :, :]))
    union = np.sum(np.logical_or(preds_occupy, gt[:, 1, :, :]))
    num_fp = np.sum(np.logical_and(preds_occupy, gt[:, 0, :, :]))  # false positive
    num_fn = np.sum(np.logical_and(np.logical_not(preds_occupy), gt[:, 1, :, :]))  # false negative
    return np.array([diff, intersection, union, num_fp, num_fn])


def voxel2mesh(voxels, surface_view):
    cube_verts = [[0, 0, 0], [0, 0, 1], [0, 1, 0], [0, 1, 1], [1, 0, 0], [1, 0, 1], [1, 1, 0],
                  [1, 1, 1]]  # 8 points

    cube_faces = [[0, 1, 2], [1, 3, 2], [2, 3, 6], [3, 7, 6], [0, 2, 6], [0, 6, 4], [0, 5, 1],
                  [0, 4, 5], [6, 7, 5], [6, 5, 4], [1, 7, 3], [1, 5, 7]]  # 12 face

    cube_verts = np.array(cube_verts)
    cube_faces = np.array(cube_faces) + 1

    scale = 0.01
    cube_dist_scale = 1.1
    verts = []
    faces = []
    curr_vert = 0

    positions = np.where(voxels > 0.3)
    voxels[positions] = 1
    for i, j, k in zip(*positions):
        # identifies if current voxel has an exposed face
        if not surface_view or np.sum(voxels[i - 1:i + 2, j - 1:j + 2, k - 1:k + 2]) < 27:
            verts.extend(scale * (cube_verts + cube_dist_scale * np.array([[i, j, k]])))
            faces.extend(cube_faces + curr_vert)
            curr_vert += len(cube_verts)

    return np.array(verts), np.array(faces)


def write_obj(filename, verts, faces):
    """ write the verts and faces on file."""
    with open(filename, 'w') as f:
        # write vertices
        f.write('g\n# %d vertex\n' % len(verts))
        for vert in verts:
            f.write('v %f %f %f\n' % tuple(vert))

        # write faces
        f.write('# %d faces\n' % len(faces))
        for face in faces:
            f.write('f %d %d %d\n' % tuple(face))


def voxel2obj(filename, pred, surface_view=True):
    verts, faces = voxel2mesh(pred, surface_view)
    write_obj(filename, verts, faces)
