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
import lib.binvox_rw as binvox_rw
import pdb

from lib.solver import save_image
from skimage.transform import rescale, resize, downscale_local_mean

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

def calculate_border(obj):
	ret = np.zeros(np.shape(obj))
	r,c,h = np.shape(ret)
	for i in range(r):
		for j in range(c):
			for k in range(h):
				if i < 10 or j < 10 or k < 10:
					ret[i,j,k] = 1
				elif i + 10 > r or j+10> c or k+10>h:
					ret[i,j,k] = 1
	return ret
	# print('shape: ', np.shape(obj))
	# ret = np.ones(np.shape(obj))
	# r,c,h = np.shape(obj)
	# for i in range(r):
	# 	for j in range(c):
	# 		for k in range(h):
	# 			if obj[i,j,k]:
	# 				border = False
	# 				if is_boundary((i,j,k), np.shape(obj)):
	# 					border = True
	# 				elif obj[i-1,j,k] == 0:
	# 					border = True
	# 				elif obj[i+1, j, k] == 0:
	# 					border = True
	# 				elif obj[i,j-1,k] == 0:
	# 					border = True
	# 				elif obj[i,j+1,k] == 0:
	# 					border = True
	# 				elif obj[i,j,k-1] == 0:
	# 					border = True
	# 				elif obj[i,j,k+1] == 0:
	# 					border = True
	# 				if border == False:
	# 					ret[i,j,k] = 0
	# 					print('changed')
	# return ret

def main():
	'''Main demo function'''
	# Save prediction into a file named 'prediction.obj' or the given argument
	#pred_file_name = sys.argv[1] if len(sys.argv) > 1 else 'prediction.obj'
	pred_file_name = 'prediction.obj'
	
	# load images
	demo_imgs = load_images('chair')
	with open(sys.argv[1], 'rb') as f:
		model = binvox_rw.read_as_3d_array(f)
	groundtruth_model = model.data
	download_model(DEFAULT_WEIGHTS)
	# Use the default network model
	NetClass = load_model('ResidualGRUNet')
	# Define a network and a solver. Solver provides a wrapper for the test function.
	net = NetClass(compute_grad=False)  # instantiate a network

	net.load(DEFAULT_WEIGHTS)                        # load downloaded weights
	solver = Solver(net)                # instantiate a solver

	flow = np.zeros((demo_imgs.shape[0], cfg.CONST.BATCH_SIZE, 2, demo_imgs.shape[-2], demo_imgs.shape[-1])).astype(theano.config.floatX)
	target,_ = solver.test_output(demo_imgs, flow=flow)
	voxel2obj('target.obj', target[0, :, 1, :, :] > cfg.TEST.VOXEL_THRESH)
	
	voxel2obj('groundtruth.obj', groundtruth_model)
	
	#print("Active target set", (groundtruth_model ^ target[0, :, 1, :, :]).sum())

	# # Save the prediction to an OBJ file (mesh file).
	# voxel2obj(pred_file_name, voxel_prediction[0, :, 1, :, :] > cfg.TEST.VOXEL_THRESH)
	#
	# # Use meshlab or other mesh viewers to visualize the prediction.
	# # For Ubuntu>=14.04, you can install meshlab using
	# # `sudo apt-get install meshlab`
	# if cmd_exists('meshlab'):
	# 	call(['meshlab', pred_file_name])
	# else:
	# 	print('Meshlab not found: please use visualization of your choice to view %s' %
	# 		  pred_file_name)


if __name__ == '__main__':
	# Set the batch size to 1
	cfg_from_list(['CONST.BATCH_SIZE', 1])
	main()
