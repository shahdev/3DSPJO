import os
import theano
import numpy as np
from lib.config import cfg
from matplotlib import pyplot as plt
from numpy import linalg as LA

from lib.voxel import voxel2obj


def max_or_nan(params):
	for param_idx, param in enumerate(params):
		# If there is nan, max will return nan
		nan_or_max_param = np.max(np.abs(param.val.get_value()))
		print('param %d : %f' % (param_idx, nan_or_max_param))
	return nan_or_max_param


def save_image(img, file_name):
	fig = plt.figure(figsize=(75, 75))  # width, height in inches

	for i in range(img.shape[0]):
		sub = fig.add_subplot(img.shape[0], 1, i + 1)
		sub.imshow(np.transpose(img[i, 0, :, :, :], (1, 2, 0)), interpolation='nearest')
		fig.savefig(file_name)
	plt.close()


def get_foreground(imgs):
	foreground_mask = (imgs < 0.99)
	foreground_mask = foreground_mask.astype(float)
	return foreground_mask

def SGD(lr, params, grads, loss):
	"""
	Stochastic Gradient Descent w/ momentum
	"""
	momentum = cfg.TRAIN.MOMENTUM
	w_decay = cfg.TRAIN.WEIGHT_DECAY

	updates = []
	for param, grad in zip(params, grads):
		vel = theano.shared(param.val.get_value() * 0.)

		if param.is_bias or w_decay == 0:
			regularized_grad = grad
		else:
			regularized_grad = grad + w_decay * param.val

		param_additive = momentum * vel - lr * regularized_grad
		updates.append((vel, param_additive))
		updates.append((param.val, param.val + param_additive))

	return updates

class Solver(object):

	def __init__(self, net):
		self.net = net
		self._test_output = None

	def compile_variables(self):
		net = self.net
		lr = self.lr
		updates = SGD(lr, net.params, net.grads, net.loss)
		self.updates = updates

	def save(self, training_losses, save_dir, step):
		''' Save the current network parameters to the save_dir and make a
		symlink to the latest param so that the training function can easily
		load the latest model'''
		save_path = os.path.join(save_dir, 'weights.%d' % (step))
		self.net.save(save_path)

		# Make a symlink for weights.npy
		symlink_path = os.path.join(save_dir, 'weights.npy')
		if os.path.lexists(symlink_path):
			os.remove(symlink_path)

		# Make a symlink to the latest network params
		os.symlink("%s.npy" % os.path.abspath(save_path), symlink_path)

		# Write the losses
		with open(os.path.join(save_dir, 'loss.%d.txt' % step), 'w') as f:
			f.write('\n'.join([str(l) for l in training_losses]))

	def is_boundary(self, pos, shape):
		return pos[0] == 0 or pos[1] == 0 or pos[2] == 0 or pos[0] == shape[0] - 1 or pos[1] == shape[1] - 1 \
			   or pos[2] == shape[2] - 1

	def calculate_border_mask(self, obj, weight):
		print('shape: ', np.shape(obj))
		ret = np.ones(np.shape(obj))
		r, c, h = np.shape(obj)
		for i in range(r):
			for j in range(c):
				for k in range(h):
					if obj[i, j, k]:
						border = False
						if self.is_boundary((i, j, k), np.shape(obj)):
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
							ret[i, j, k] = 1
						else:
							ret[i, j, k] = weight
					else:
						ret[i, j, k] = weight
		return ret

	def get_image(self, images, flows):
		[batch_size, _, H, W] = images.shape

		basegrid = np.stack(np.meshgrid(np.arange(H), np.arange(W),
										indexing='ij')).astype(theano.config.floatX)
		batched_basegrid = np.tile([basegrid], [batch_size, 1, 1, 1])

		sampling_grid = batched_basegrid + flows

		sampling_grid_x = np.clip(
			sampling_grid[:, 1], 0., np.cast[theano.config.floatX](W - 1)
		)
		sampling_grid_y = np.clip(
			sampling_grid[:, 0], 0., np.cast[theano.config.floatX](H - 1)
		)

		images = np.transpose(images, [0, 2, 3, 1])
		# now we need to interpolate

		# grab 4 nearest corner points for each (x_i, y_i)
		# i.e. we need a square around the point of interest
		x0 = np.cast['int32'](np.floor(sampling_grid_x))
		x1 = x0 + 1
		y0 = np.cast['int32'](np.floor(sampling_grid_y))
		y1 = y0 + 1

		x0 = np.clip(x0, 0, W - 2)
		x1 = np.clip(x1, 0, W - 1)
		y0 = np.clip(y0, 0, H - 2)
		y1 = np.clip(y1, 0, H - 1)

		# b is a (B, H, W) tensor with (B, H, W) = B for all (H, W)
		b = np.tile(
			np.reshape(
				np.arange(0, batch_size), (batch_size, 1, 1)
			),
			(1, H, W)
		)

		# arr = np.stack([b, y0, x0], 3)
		arr = [b, y0, x0]
		# Ia = np.take(images, np.ravel_multi_index(arr, (batch_size, H, W, 3)))
		Ia = np.zeros((batch_size, H, W, 3)).astype(theano.config.floatX)
		Ia[:, :, :, 0] = np.take(images[:, :, :, 0], np.ravel_multi_index(arr, (batch_size, H, W)))
		Ia[:, :, :, 1] = np.take(images[:, :, :, 1], np.ravel_multi_index(arr, (batch_size, H, W)))
		Ia[:, :, :, 2] = np.take(images[:, :, :, 2], np.ravel_multi_index(arr, (batch_size, H, W)))

		# arr = np.stack([b, y1, x0], 3)
		arr = [b, y1, x0]
		Ib = np.zeros((batch_size, H, W, 3)).astype(theano.config.floatX)
		Ib[:, :, :, 0] = np.take(images[:, :, :, 0], np.ravel_multi_index(arr, (batch_size, H, W)))
		Ib[:, :, :, 1] = np.take(images[:, :, :, 1], np.ravel_multi_index(arr, (batch_size, H, W)))
		Ib[:, :, :, 2] = np.take(images[:, :, :, 2], np.ravel_multi_index(arr, (batch_size, H, W)))

		# arr = np.stack([b, y0, x1], 3)
		arr = [b, y0, x1]
		Ic = np.zeros((batch_size, H, W, 3)).astype(theano.config.floatX)
		Ic[:, :, :, 0] = np.take(images[:, :, :, 0], np.ravel_multi_index(arr, (batch_size, H, W)))
		Ic[:, :, :, 1] = np.take(images[:, :, :, 1], np.ravel_multi_index(arr, (batch_size, H, W)))
		Ic[:, :, :, 2] = np.take(images[:, :, :, 2], np.ravel_multi_index(arr, (batch_size, H, W)))

		# arr = np.stack([b, y1, x1], 3)
		arr = [b, y1, x1]
		Id = np.zeros((batch_size, H, W, 3)).astype(theano.config.floatX)
		Id[:, :, :, 0] = np.take(images[:, :, :, 0], np.ravel_multi_index(arr, (batch_size, H, W)))
		Id[:, :, :, 1] = np.take(images[:, :, :, 1], np.ravel_multi_index(arr, (batch_size, H, W)))
		Id[:, :, :, 2] = np.take(images[:, :, :, 2], np.ravel_multi_index(arr, (batch_size, H, W)))

		x0 = np.cast[theano.config.floatX](x0)
		x1 = np.cast[theano.config.floatX](x1)
		y0 = np.cast[theano.config.floatX](y0)
		y1 = np.cast[theano.config.floatX](y1)

		# calculate deltas
		wa = (x1 - sampling_grid_x) * (y1 - sampling_grid_y)
		wb = (x1 - sampling_grid_x) * (sampling_grid_y - y0)
		wc = (sampling_grid_x - x0) * (y1 - sampling_grid_y)
		wd = (sampling_grid_x - x0) * (sampling_grid_y - y0)

		# add dimension for addition
		wa = np.expand_dims(wa, axis=3)
		wb = np.expand_dims(wb, axis=3)
		wc = np.expand_dims(wc, axis=3)
		wd = np.expand_dims(wd, axis=3)

		perturbed_image = wa * Ia + wb * Ib + wc * Ic + wd * Id

		perturbed_image = np.transpose(perturbed_image, [0, 3, 1, 2])

		return perturbed_image

	def spatial_dag_attack(self, x, flow_init, save_directory, target=None, max_iters=10, attack_epsilon=1/255, tau=0.0, alpha_flow=0.1,
						   alpha_inp=0.1, source_img=None, target_img=None, background_attack=True, weight=2,
						   dag_only=False, spatial_only=False):

		directory = save_directory		

		if not os.path.exists(directory):
			os.makedirs(directory)

		output = theano.function([self.net.x, self.net.flow],
								 [self.net.output])
		print(np.shape(x))

		grad_wrt_inp = theano.function([self.net.x, self.net.flow, self.net.y, self.net.mask, self.net.tau],
									   [theano.tensor.grad(self.net.loss, self.net.x),
										theano.tensor.grad(self.net.flow_loss, self.net.flow), self.net.loss,
										self.net.flow_loss, self.net.reg_term, self.net.tv_loss])
		n_vox = cfg.CONST.N_VOX
		mu = 0.85
		if target is None:
			target = np.zeros(
				(cfg.CONST.BATCH_SIZE, n_vox, 2, n_vox, n_vox))
			target[:, :, 1, :, :] = 1
			target = target.astype(theano.config.floatX)

		x_adv = x.copy()
		flow = flow_init.copy()

		pgd_max = x + attack_epsilon
		pgd_min = x - attack_epsilon

		log_file = open(directory + "log.txt", "w")

		y_val = target
		target = target[0, :, 1, :, :] > cfg.TEST.VOXEL_THRESH
		print('target shape: ', np.shape(target))
		results = output(x_adv, flow)
		prediction = results[0]
		predictions = prediction[0, :, 1, :, :] > cfg.TEST.VOXEL_THRESH

		boundary_mask = self.calculate_border_mask(target, weight)
		active_targets = predictions ^ target
		mask = active_targets * boundary_mask  # * self.weighted_mask(active_targets)
		mask = np.asarray([mask] * cfg.CONST.BATCH_SIZE)
		mask = np.asarray([mask] * 2)
		mask = np.transpose(mask, (1, 2, 0, 3, 4)).astype(theano.config.floatX)
		iter_ = 0
		grad_inp_t = np.zeros(x_adv.shape).astype(theano.config.floatX)
		grad_flow_t = np.zeros(flow.shape).astype(theano.config.floatX)

		background_mask = 1 - get_foreground(x)

		while active_targets.sum() > 0 and iter_ < max_iters:
			print('active target length: ', active_targets.sum())
			iou = (predictions & target).sum() / (predictions | target).sum()
			print('iou: ', iou)
			previous_targets = active_targets.sum()
			print('Iteration: ', iter_)
			grad = grad_wrt_inp(x_adv, flow, y_val, mask, tau)
			grad_inp = grad[0]
			grad_flow = grad[1]
			tv_loss = grad[5]
			print('tv loss: ', tv_loss)
			# grad_x = lb*(x_adv - x) + grad_inp
			grad_x = grad_inp

			if iter_ == 0:
				grad_inp_t = grad_x
				grad_flow_t = grad_flow
			else:
				grad_inp_t = mu * grad_inp_t + (1 - mu) * grad_x
				grad_flow_t = mu * grad_flow_t + (1 - mu) * grad_flow

			foreground_mask = get_foreground(x_adv)

			if dag_only:
				x_adv = np.clip(x_adv - alpha_inp * grad_inp, 0.0, 1.0)
				x_adv = np.clip(x_adv - pgd_max, None, 0) + pgd_max
				x_adv = np.clip(x_adv - pgd_min, 0, None) + pgd_min

				if not background_attack:
					x_adv = np.float32(foreground_mask * x_adv + (1 - foreground_mask) * x)
			elif spatial_only:
				flow = flow - alpha_flow*grad_flow_t
			else:
				flow = flow - alpha_flow* grad_flow_t
				x_adv = np.clip(x_adv - alpha_inp * grad_inp, 0.0, 1.0)
				x_adv = np.clip(x_adv - pgd_max, None, 0) + pgd_max
				x_adv = np.clip(x_adv - pgd_min, 0, None) + pgd_min
				if not background_attack:
					x_adv = np.float32(foreground_mask * x_adv + (1 - foreground_mask) * x)

			print("losses : ", grad[2])
			print("flow loss: ", grad[3])

			reg_term = grad[4]
			# print('shape reg: ', np.shape(reg_term))
			# print('reg term: ', reg_term)
			print("flow data max min: %f, %f" % (np.max(flow), np.min(flow)))
			print("noise perturbation max: %f" % (np.max(np.abs(x_adv - x))))
			iter_ += 1
			if iter_ % 200 == 1:
				alpha_flow*= 1.2
				alpha_inp *= 0.8
				im0 = self.get_image(x_adv[0], flow[0])
				im1 = self.get_image(x_adv[1], flow[1])
				im2 = self.get_image(x_adv[2], flow[2])
				ims = np.array([im0, im1, im2])
				l2_loss = np.sqrt(np.sum(np.multiply(ims - x, ims - x)))
				save_image(x_adv, '%s/pixel_%d.png'%(directory, iter_))
				save_image(ims, '%s/spatial_pixel_%d.png'%(directory, iter_))
				np.save('%s/pixel_%d.npy'%(directory, iter_), x_adv)
				np.save('%s/flow_%d.npy'%(directory, iter_), flow)
				print('savng prediction')
				voxel2obj('%s/voxel_%d.obj'%(directory, iter_), predictions)

				print("IOU : %f \n" % iou)
				print("Misclassified voxels : %d \n" % active_targets[index].sum())
				print("Accuracy : %f \n" % (1 - (active_targets[index].sum()) / 32768.0))

				log_file.write("IOU : %f \n" % iou)
				log_file.write("Misclassified voxels : %d \n" % active_targets[index].sum())
				log_file.write("Accuracy : %f \n" % (1 - (active_targets[index].sum()) / 32768.0))

			results = output(x_adv, flow)
			prediction = results[0]
			predictions = prediction[0, :, 1, :, :] > cfg.TEST.VOXEL_THRESH
			active_targets = predictions ^ target
			mask = active_targets * boundary_mask  # * self.weighted_mask(active_targets)
			# mask = self.calculate_border(active_targets)
			mask = np.asarray([mask] * cfg.CONST.BATCH_SIZE)
			mask = np.asarray([mask] * 2)
			mask = np.transpose(mask, (1, 2, 0, 3, 4)).astype(theano.config.floatX)

		log_file.close()
		return prediction, None

	def test_output(self, x, y=None, mask=None, flow=None):
		'''Generate the reconstruction, loss, and activation. Evaluate loss if
		ground truth output is given. Otherwise, return reconstruction and
		activation'''
		# Cache the output function.
		if mask is None:
			mask = np.ones((cfg.CONST.BATCH_SIZE, 32, 2, 32, 32)).astype(theano.config.floatX)

		if self._test_output is None:
			print('Compiling testing function')
			# Lazy load the test function
			self._test_output = theano.function([self.net.x, self.net.y, self.net.mask, self.net.flow],
												[self.net.output,
												 self.net.loss,
												 *self.net.activations])
		# If the ground truth data is given, evaluate loss. O.w. feed zeros and
		# does not return the loss
		if y is None:
			n_vox = cfg.CONST.N_VOX
			no_loss_return = True
			y_val = np.zeros(
				(cfg.CONST.BATCH_SIZE, n_vox, 2, n_vox, n_vox)).astype(theano.config.floatX)
			mask = np.ones((cfg.CONST.BATCH_SIZE, 32, 2, 32, 32)).astype(theano.config.floatX)
			results = self._test_output(x, y_val, mask, flow)
			prediction = results[0]
			loss = results[1]
			activations = results[2:]
			return prediction, activations
			y = results[0]

		no_loss_return = True
		y_val = y

		# self.grad_wrt_inp = theano.function([self.net.x, self.net.y],
		#                                     [theano.tensor.grad(self.net.loss, self.net.x)])

		# res = self.grad_wrt_inp(x,y_val)
		# np.save('grad', res)
		# print(res)

		# Parse the result
		print('run')
		results = self._test_output(x, y_val, mask, flow)
		print('ran')
		prediction = results[0]
		loss = results[1]
		activations = results[2:]

		if no_loss_return:
			return prediction, activations
		else:
			return prediction, loss, activations
