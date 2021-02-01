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

from torch.autograd import Variable
from PIL import Image
import matplotlib.pyplot as plt
import scipy.misc
import sys
import torch.nn as nn
import torch.nn.functional as F


def load_images(obj):
	ims = []
	for i in range(3):
		im = Image.open('imgs/' + obj + str(i) + '.png')
		ims.append([np.array(im)[:, :, 0:3].transpose(
			(2, 0, 1)).astype(np.float32) / 255.])
	return np.array(ims)


def is_boundary(pos, shape):
	return pos[0] == 0 or pos[1] == 0 or pos[2] == 0 or pos[0] == shape[0] - 1 or pos[1] == shape[1] - 1 \
		   or pos[2] == shape[2] - 1


def calculate_border_mask(obj, weight):
	ret = np.ones(np.shape(obj))
	r, c, h = np.shape(obj)
	for i in range(r):
		for j in range(c):
			for k in range(h):
				if obj[i, j, k]:
					border = False
					if is_boundary((i, j, k), np.shape(obj)):
						border = True
					elif obj[i - 1, j, k] == 0:
						border = True
					elif obj[i + 1, j, k] == 0:
						border = True
					elif obj[i, j - 1, k] == 0:
						border = True
					elif obj[i, j + 1, k] == 0:
						border = True
					elif obj[i, j, k - 1] == 0:
						border = True
					elif obj[i, j, k + 1] == 0:
						border = True

					if border == False:
						ret[i, j, k] = 1.0 / weight
					else:
						ret[i, j, k] = 1.0
				else:
					ret[i, j, k] = 1.0
	return ret


def calculate_external_voxels_mask(obj):
	ret = np.ones(np.shape(obj), dtype=bool)
	r, c, h = np.shape(obj)
	for i in range(r):
		for j in range(c):
			for k in range(h):
				if obj[i, j, k]:
					border = False
					if is_boundary((i, j, k), np.shape(obj)):
						border = True
					elif obj[i - 1, j, k] == 0:
						border = True
					elif obj[i + 1, j, k] == 0:
						border = True
					elif obj[i, j - 1, k] == 0:
						border = True
					elif obj[i, j + 1, k] == 0:
						border = True
					elif obj[i, j, k - 1] == 0:
						border = True
					elif obj[i, j, k + 1] == 0:
						border = True
					ret[i, j, k] = border
				else:
					ret[i, j, k] = True
	return ret


def save_images(imgs, file_name):
	plt.axis('off')
	# import pdb
	# pdb.set_trace()
	for i in range(imgs.shape[0]):
		img = imgs[i]
		plt.imsave(file_name + str(i) + '.png', np.clip(np.transpose(img, (1, 2, 0)), 0.0, 1.0))


# plt.imsave(np.clip(np.transpose(img, (1,2,0)), 0.0, 1.0))
# scipy.misc.imsave(file_name + str(i) + '.png', np.clip(np.transpose(img, (1,2,0)), 0.0, 1.0))
# plt.savefig(file_name + str(i) + '.png')
# plt.close(fig)


def save_flow(flow, directory):
	X_ = np.arange(0, 224, 1)
	Y_ = np.arange(223, -1, -1)

	U = flow.data.cpu().numpy()

	for i in range(U.shape[0]):
		fig, ax = plt.subplots()
		plt.axis('off')
		q = ax.quiver(X_, Y_, U[i, :, :, 0], U[i, :, :, 1])
		plt.savefig(directory + 'flow_%d.png' % i)


class Loss_flow(nn.Module):
	def __init__(self, neighbours=np.array([[1, 1, 1], [1, 0, 1], [1, 1, 1]])):
		super(Loss_flow, self).__init__()

		filters = []
		for i in range(neighbours.shape[0]):
			for j in range(neighbours.shape[1]):
				if neighbours[i][j] == 1:
					filter = np.zeros((1, neighbours.shape[0], neighbours.shape[1]))
					filter[0][i][j] = -1
					filter[0][neighbours.shape[0] // 2][neighbours.shape[1] // 2] = 1
					filters.append(filter)

		filters = np.array(filters)
		self.filters = torch.from_numpy(filters).float()

	def forward(self, f):
		# TODO: paddingfilename
		'''
		f - f.size() =  [1, h, w, 2]
			f[0, :, :, 0] - u channel
			f[0, :, :, 1] - v channel
		'''
		f_u = f[:, :, :, 0].unsqueeze(1)
		f_v = f[:, :, :, 1].unsqueeze(1)

		diff_u = F.conv2d(f_u, self.filters)[0][0]  # don't use squeeze
		diff_u_sq = torch.mul(diff_u, diff_u)

		diff_v = F.conv2d(f_v, self.filters)[0][0]  # don't use squeeze
		diff_v_sq = torch.mul(diff_v, diff_v)

		dist = torch.sqrt(torch.sum(diff_u_sq, dim=0) + torch.sum(diff_v_sq, dim=0))
		return torch.sum(dist) / (dist.shape[0] * dist.shape[0])

	def to(self, device=None):
		self = super().to(device=device)
		self.filters = self.filters.to(device=device)
		return self


def get_foreground(imgs):
	foreground_mask = 1.0*(imgs < 0.99)
	return foreground_mask


def test_net(cfg,
			 epoch_idx=-1,
			 output_dir=None,
			 test_data_loader=None,
			 test_writer=None,
			 encoder=None,
			 decoder=None,
			 merger=None,
			 sources=None,
			 targets=None,
			 weight=10,
			 refiner=None,
			 args=None):
	# Enable the inbuilt cudnn auto-tuner to find the best algorithm to use
	torch.backends.cudnn.benchmark = True
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	print(device)
	source_imgs = []
	sources = sources[:10]
	targets = targets[:10]

	# alpha_inp = 0.0020
	alpha_inp = 0.010
	# alpha_inp = 1
	alpha_flow = 0.0020
	iter_ = 0
	max_iters = 25000
	attack_epsilon = 8.0 / 25
	threshold = 0.4
	mu = 0.85
	lb = 1
	tau = 10

	if args is not None:
		alpha_inp = args.alpha_inp
		alpha_flow = args.alpha_flow
		attack_epsilon = args.attack_epsilon / 255
		tau = args.tau

	log_files = []
	for index in range(len(sources)):
		if args.foreground_only == 1:
			directory = args.attack_type + '_foreground/' + 'weight_%d/' % weight + 'epsilon_%d' % args.attack_epsilon + '/' + \
						sources[index] + '_' + targets[index] + '/'
		else:
			directory = args.attack_type + '/' + 'weight_%d/' % weight + 'epsilon_%d' % args.attack_epsilon + '/' + \
						sources[index] + '_' + targets[index] + '/'
		if not os.path.exists(directory):
			os.makedirs(directory)

		log_file = open(directory + "log.txt", "w")
		log_files.append(log_file)

	for source_ in sources:
		demo_imgs = load_images(source_)
		demo_imgs -= cfg.DATASET.MEAN[0]
		demo_imgs /= cfg.DATASET.STD[0]
		demo_imgs = np.transpose(demo_imgs, (1, 0, 2, 3, 4))
		source_imgs.append(demo_imgs)

	source_imgs = np.vstack(source_imgs)
	source_imgs = torch.tensor(source_imgs, device=device).float()
	source_imgs = Variable(source_imgs, requires_grad=True)

	clean_imgs = source_imgs.clone().detach()

	assert (len(sources) == len(targets))
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
		else:
			encoder = torch.nn.DataParallel(encoder)
			decoder = torch.nn.DataParallel(decoder)
			refiner = torch.nn.DataParallel(refiner)
			merger = torch.nn.DataParallel(merger)

		print('[INFO] %s Loading weights from %s ...' % (dt.now(), cfg.CONST.WEIGHTS))
		checkpoint = torch.load(cfg.CONST.WEIGHTS, map_location=device)
		epoch_idx = checkpoint['epoch_idx']
		encoder.load_state_dict(checkpoint['encoder_state_dict'])
		decoder.load_state_dict(checkpoint['decoder_state_dict'])

		if cfg.NETWORK.USE_REFINER:
			refiner.load_state_dict(checkpoint['refiner_state_dict'])
		if cfg.NETWORK.USE_MERGER:
			merger.load_state_dict(checkpoint['merger_state_dict'])

	# Set up loss functions
	bce_loss = torch.nn.BCELoss()
	mse_loss = torch.nn.MSELoss()
	loss_flow = Loss_flow()
	loss_flow.to(device=device)
	# Switch models to evaluation mode
	encoder.eval()
	decoder.eval()
	refiner.eval()
	merger.eval()

	target_imgs = []
	for target_ in targets:
		target_img = load_images(target_)
		target_img -= cfg.DATASET.MEAN[0]
		target_img /= cfg.DATASET.STD[0]
		target_img = np.transpose(target_img, (1, 0, 2, 3, 4))
		target_imgs.append(target_img)

	target_imgs = np.vstack(target_imgs)
	target_imgs = torch.tensor(target_imgs, device=device).float()

	target_image_features = encoder(target_imgs)
	target_raw_features, target_generated_volume = decoder(target_image_features)

	if cfg.NETWORK.USE_MERGER and epoch_idx >= cfg.TRAIN.EPOCH_START_USE_MERGER:
		target_generated_volume = merger(target_raw_features, target_generated_volume)
	else:
		target_generated_volume = torch.mean(target_generated_volume, dim=1)

	if cfg.NETWORK.USE_REFINER and epoch_idx >= cfg.TRAIN.EPOCH_START_USE_REFINER:
		target_generated_volume = refiner(target_generated_volume)

	ground_truth_volume = target_generated_volume.detach()
	_target = torch.ge(ground_truth_volume, threshold)
	boundary_mask = []
	for index in range(ground_truth_volume.shape[0]):
		boundary_mask.append(calculate_border_mask(_target[index].cpu().detach().numpy(), weight))
	boundary_mask = torch.tensor(boundary_mask, device=device)
	# boundary_mask = torch.tensor(boundary_mask, device=device)

	# Test the encoder, decoder and merger

	grad_inp_t = torch.zeros(source_imgs.shape, device=device)

	source_imgs_collate = source_imgs.view(source_imgs.shape[0] * source_imgs.shape[1], source_imgs.shape[2],
										   source_imgs.shape[3], source_imgs.shape[4])

	batch_size = source_imgs.shape[0]
	theta = torch.tensor([[1, 0, 0], [0, 1, 0]], device=device).float()  # identity transformation
	n_views = source_imgs.shape[1]
	theta = theta.repeat(n_views * batch_size, 1, 1)
	grid = F.affine_grid(theta, source_imgs_collate.shape, align_corners=True)
	# grid.size() = (N_VIEWS*BATCH_SIZE, h, w, 2)

	f = Variable(torch.zeros_like(grid).float(), requires_grad=True)
	torch.nn.init.normal_(f, mean=0, std=0.00001)

	grad_flow_t = torch.zeros(f.shape, device=device)

	pgd_max = clean_imgs + attack_epsilon
	pgd_min = clean_imgs - attack_epsilon
	spatial_counter = 0
	while iter_ < max_iters:
		source_imgs_collate = source_imgs_collate.cpu()
		del source_imgs_collate
		source_imgs.requires_grad_()
		f.requires_grad_()
		with torch.enable_grad():
			grid_new = grid + f
			grid_new = grid_new.clamp(min=-1, max=1)
			source_imgs_collate = source_imgs.view(source_imgs.shape[0] * source_imgs.shape[1], source_imgs.shape[2],
												   source_imgs.shape[3], source_imgs.shape[4])
			x_new = F.grid_sample(source_imgs_collate, grid_new, mode='bilinear', align_corners=True)
			x_new = x_new.view(source_imgs.shape[0], source_imgs.shape[1], source_imgs.shape[2],
							   source_imgs.shape[3], source_imgs.shape[4])

			image_features = encoder(x_new)

			# image_features = encoder(rendering_images)
			raw_features, generated_volume = decoder(image_features)

			if cfg.NETWORK.USE_MERGER and epoch_idx >= cfg.TRAIN.EPOCH_START_USE_MERGER:
				generated_volume = merger(raw_features, generated_volume)
			else:
				generated_volume = torch.mean(generated_volume, dim=1)

			_volume = torch.ge(generated_volume, threshold)
			active_targets = _volume ^ _target
			# print('iter: ', iter_, 'active target length: ', active_targets.sum())
			mask = active_targets * 1.0 * boundary_mask
			# encoder_loss = bce_loss(generated_volume * mask, ground_truth_volume * mask) * 10 * weight
			encoder_loss = torch.mean(-(
					ground_truth_volume * torch.log(generated_volume + 1e-9) + (1 - ground_truth_volume) * torch.log(
				1 - generated_volume + 1e-9)) * mask) * 10
			if cfg.NETWORK.USE_REFINER and epoch_idx >= cfg.TRAIN.EPOCH_START_USE_REFINER:
				generated_volume = refiner(generated_volume)
			else:
				refiner_loss = encoder_loss

			_volume = torch.ge(generated_volume, threshold)
			active_targets = _volume ^ _target
			# print('iter: ', iter_, 'active target length: ', active_targets.sum())
			mask = active_targets * 1.0 * boundary_mask
			# refiner_loss = bce_loss(generated_volume * mask, ground_truth_volume * mask) * 10 * weight
			refiner_loss = torch.mean(-(
					ground_truth_volume * torch.log(generated_volume + 1e-9) + (1 - ground_truth_volume) * torch.log(
				1 - generated_volume + 1e-9)) * mask) * 10
			l2_loss = mse_loss(clean_imgs, source_imgs)
			linf_loss = torch.max(clean_imgs - source_imgs)
			flow_regularization_loss = loss_flow(f)
			# print(iter_, active_targets.data.sum(), encoder_loss.data, refiner_loss.data, l2_loss.data, '{0:.10f}'.format(flow_regularization_loss.data), linf_loss.data, flush=True)
			total_loss = 0.5 * (encoder_loss + refiner_loss) + tau * flow_regularization_loss

		# encoder.zero_grad()
		# decoder.zero_grad()
		# merger.zero_grad()
		# refiner.zero_grad()
		#
		# if source_imgs.grad is not None:
		# 	source_imgs.grad.zero()
		# 	f.grad.zero()

		#total_loss.backward()
		grad_inp, grad_flow = torch.autograd.grad(outputs=total_loss, inputs=[source_imgs, f], grad_outputs=[torch.Tensor([1., 1.])], only_inputs=True)

		# grad_inp = source_imgs.grad
		# grad_flow = f.grad

		print(iter_, active_targets.data.sum(), torch.norm(grad_inp).data, torch.norm(grad_flow).data, f.data.max(),
			  f.data.min(), flush=True)

		grad_flow_t = mu * grad_flow_t + (1 - mu) * grad_flow.detach()  # / torch.norm(grad_flow)
		grad_inp_t = mu * grad_inp_t + (1 - mu) * grad_inp.detach()  # / torch.norm(grad_inp)

		if args.attack_type == 'spatial_dag':
			if spatial_counter % 2 == 0:
				f = f.detach() - alpha_flow * grad_flow_t
			else:
				source_imgs = torch.clamp(source_imgs.detach() - alpha_inp * grad_inp_t, -1.0, 1.0)
				source_imgs = torch.clamp(source_imgs - pgd_max, max=0) + pgd_max
				source_imgs = torch.clamp(source_imgs - pgd_min, min=0) + pgd_min
		elif args.attack_type == 'dag':
			source_imgs = torch.clamp(source_imgs.detach() - alpha_inp * grad_inp_t, -1.0, 1.0)
			pgd_max = clean_imgs + attack_epsilon
			pgd_min = clean_imgs - attack_epsilon
			source_imgs = torch.clamp(source_imgs - pgd_max, max=0) + pgd_max
			source_imgs = torch.clamp(source_imgs - pgd_min, min=0) + pgd_min
		elif args.attack_type == 'spatial':
			f = f.detach() - alpha_flow * grad_flow_t

		if args.foreground_only == 1:
			foreground_mask = get_foreground(x_new.detach())
			source_imgs = foreground_mask * source_imgs + (1 - foreground_mask) * clean_imgs

		if args.attack_type == 'spatial_dag':
			spatial_counter += 1

		if spatial_counter % 2 == 0:
			iter_ += 1
		if spatial_counter % 2 == 0 and iter_ % 1000 == 999:
			lb /= 2
			alpha_flow *= 0.8
			alpha_inp *= 0.8
			x_adv = source_imgs.data.cpu().numpy()
			x_adv *= cfg.DATASET.STD[0]
			x_adv += cfg.DATASET.MEAN[0]

			x_save = x_new.data.cpu().numpy()
			x_save *= cfg.DATASET.STD[0]
			x_save += cfg.DATASET.MEAN[0]

			for index in range(len(sources)):
				# directory = args.attack_type + '/' + sources[index] + '_' + targets[index] + '/'
				if args.foreground_only == 1:
					directory = args.attack_type + '_foreground/' + 'weight_%d/' % weight + 'epsilon_%d' % args.attack_epsilon + '/' + \
							sources[index] + '_' + targets[index] + '/'
				else:
					directory = args.attack_type + '/' + 'weight_%d/' % weight + 'epsilon_%d' % args.attack_epsilon + '/' + \
							sources[index] + '_' + targets[index] + '/'
				save_images(x_adv[index], directory + 'pixel_%d_' % (iter_))
				save_images(x_save[index], directory + 'spatial_pixel_%d_' % (iter_))

				_volume = torch.ge(generated_volume[index], threshold)
				np_obj = _volume.data.cpu().data.numpy()
				voxel2obj(directory + 'prediction_%d.obj' % iter_, np_obj)

				np_obj_s = _target[index].data.cpu().data.numpy()
				iou = (np_obj & np_obj_s).sum() / (np_obj | np_obj_s).sum()
				target_boundary_mask = calculate_external_voxels_mask(_target[index].cpu().detach().numpy())
				iou_3D = (np_obj & target_boundary_mask & np_obj_s).sum() / (
						(np_obj | np_obj_s) & target_boundary_mask).sum()

				l2_source_target = mse_loss(clean_imgs[index], source_imgs[index])
				print("SOURCE : %s, TARGET: %s" % (sources[index], targets[index]))
				print("IOU : %f \n" % iou)
				print("Misclassified voxels : %d \n" % active_targets[index].sum())
				print("Accuracy : %f \n" % (1 - (active_targets[index].sum()) / 32768.0))
				print("Mean Square Loss : %f \n" % l2_source_target)
				print("IOU 3D: %f \n" % iou_3D)
				log_files[index].write("IOU : %f \n" % iou)
				log_files[index].write("Misclassified voxels : %d \n" % active_targets[index].sum())
				log_files[index].write("Accuracy : %f \n" % (1 - (active_targets[index].sum()) / 32768.0))
				log_files[index].write("Mean Square Loss : %f \n" % l2_source_target.data)

	# save_flow(f, directory)

	for index in range(len(sources)):
		log_files[index].close()


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
