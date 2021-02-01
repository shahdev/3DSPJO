import os
import sys
import theano
import theano.tensor as T
import numpy as np
from datetime import datetime

from lib.config import cfg
from lib.utils import Timer
import pdb
# %matplotlib inline
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
	foreground_mask = (imgs<0.99)
	foreground_mask=foreground_mask.astype(float)
	return foreground_mask

def ADAM(lr, params, grads, loss, iteration, beta_1=0.9, beta_2=0.999, epsilon=1e-8):
	"""
	ADAM update
	"""
	t = iteration
	lr_t = lr * T.sqrt(1 - T.pow(beta_2, t)) / (1 - T.pow(beta_1, t))
	w_decay = cfg.TRAIN.WEIGHT_DECAY

	updates = []
	for p, g in zip(params, grads):
		# zero init of moment
		m = theano.shared(p.val.get_value() * 0.)
		# zero init of velocity
		v = theano.shared(p.val.get_value() * 0.)

		if p.is_bias or w_decay == 0:
			regularized_g = g
		else:
			regularized_g = g + w_decay * p.val

		m_t = (beta_1 * m) + (1 - beta_1) * regularized_g
		v_t = (beta_2 * v) + (1 - beta_2) * T.square(regularized_g)
		p_t = p.val - lr_t * m_t / (T.sqrt(v_t) + epsilon)

		updates.append((m, m_t))
		updates.append((v, v_t))
		updates.append((p.val, p_t))

	return updates


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
		self.lr = theano.shared(np.float32(1))
		self.iteration = theano.shared(np.float32(0))  # starts from 0
		self._test = None
		self._train_loss = None
		self._test_output = None
		self.compile_model(cfg.TRAIN.POLICY)

	def compile_model(self, policy=cfg.TRAIN.POLICY):
		net = self.net
		lr = self.lr
		iteration = self.iteration

		if policy == 'sgd':
			updates = SGD(lr, net.params, net.grads, net.loss)
		elif policy == 'adam':
			updates = ADAM(lr, net.params, net.grads, net.loss, iteration)
		else:
			sys.exit('Error: Unimplemented optimization policy')

		self.updates = updates

	def set_lr(self, lr):
		self.lr.set_value(lr)

	@property
	def train_loss(self):
		if self._train_loss is None:
			print('Compiling training function')
			self._train_loss = theano.function(
				[self.net.x, self.net.y], self.net.loss, updates=self.updates, profile=cfg.PROFILE)
		self.iteration.set_value(self.iteration.get_value() + 1)
		return self._train_loss

	def train(self, train_queue, val_queue=None):
		''' Given data queues, train the network '''
		# Parameter directory
		save_dir = os.path.join(cfg.DIR.OUT_PATH)
		if not os.path.exists(save_dir):
			os.makedirs(save_dir)

		# Timer for the training op and parallel data loading op.
		train_timer = Timer()
		data_timer = Timer()
		training_losses = []

		start_iter = 0
		# Resume training
		if cfg.TRAIN.RESUME_TRAIN:
			self.net.load(cfg.CONST.WEIGHTS)
			start_iter = cfg.TRAIN.INITIAL_ITERATION

		# Setup learning rates
		lr = cfg.TRAIN.DEFAULT_LEARNING_RATE
		lr_steps = [int(k) for k in cfg.TRAIN.LEARNING_RATES.keys()]

		print('Set the learning rate to %f.' % lr)
		self.set_lr(lr)

		# Main training loop
		for train_ind in range(start_iter, cfg.TRAIN.NUM_ITERATION + 1):
			data_timer.tic()
			batch_img, batch_voxel = train_queue.get()
			data_timer.toc()

			if self.net.is_x_tensor4:
				batch_img = batch_img[0]

			# Apply one gradient step
			train_timer.tic()
			loss = self.train_loss(batch_img, batch_voxel)
			train_timer.toc()

			training_losses.append(loss)

			# Decrease learning rate at certain points
			if train_ind in lr_steps:
				# edict only takes string for key. Hacky way
				self.set_lr(np.float(cfg.TRAIN.LEARNING_RATES[str(train_ind)]))
				print('Learing rate decreased to %f: ' % self.lr.get_value())

			# Debugging modules
			#
			# Print status, run validation, check divergence, and save model.
			if train_ind % cfg.TRAIN.PRINT_FREQ == 0:
				# Print the current loss
				print('%s Iter: %d Loss: %f' % (datetime.now(), train_ind, loss))

			if train_ind % cfg.TRAIN.VALIDATION_FREQ == 0 and val_queue is not None:
				# Print test loss and params to check convergence every N iterations
				val_losses = []
				for i in range(cfg.TRAIN.NUM_VALIDATION_ITERATIONS):
					batch_img, batch_voxel = val_queue.get()
					_, val_loss, _ = self.test_output(batch_img, batch_voxel)
					val_losses.append(val_loss)
				print('%s Test loss: %f' % (datetime.now(), np.mean(val_losses)))

			if train_ind % cfg.TRAIN.NAN_CHECK_FREQ == 0:
				# Check that the network parameters are all valid
				max_param = max_or_nan(self.net.params)
				if np.isnan(max_param):
					print('NAN detected')
					break

			if train_ind % cfg.TRAIN.SAVE_FREQ == 0 and not train_ind == 0:
				self.save(training_losses, save_dir, train_ind)

			if loss > cfg.TRAIN.LOSS_LIMIT:
				print("Cost exceeds the threshold. Stop training")
				break

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
		r,c,h = np.shape(obj)
		for i in range(r):
			for j in range(c):
				for k in range(h):
					if obj[i,j,k]:
						border = False
						if self.is_boundary((i,j,k), np.shape(obj)):
							border = True
						elif obj[i-1,j,k] == 0:
							border = True
						elif obj[i+1, j, k] == 0:
							border = True
						elif obj[i,j-1,k] == 0:
							border = True
						elif obj[i,j+1,k] == 0:
							border = True
						elif obj[i,j,k-1] == 0:
							border = True
						elif obj[i,j,k+1] == 0:
							border = True

						if border == False:
							ret[i,j,k] = 1
						else:
							ret[i,j,k] = weight
					else:
						ret[i,j,k] = weight											
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

	def spatial_dag_attack(self, x, flow_init, target=None, max_iters=10, lb=10, tau=0.1, alpha=0.1,
						   alpha_inp=0.1, source_img=None, target_img=None, background_attack=True, weight=2, dag_only=False, spatial_only=False):
		
		if dag_only:
			if background_attack:
				directory = 'DAG/background_'+source_img+'_'+target_img + str(lb)+'_'+str(tau)
			else:
				directory = 'DAG/foreground_'+source_img+'_'+target_img + str(lb)+'_'+str(tau)
		elif spatial_only:
			if background_attack:
				directory = 'SPATIAL/background_'+source_img+'_'+target_img + str(lb)+'_'+str(tau)
			else:
				directory = 'SPATIAL/foreground_'+source_img+'_'+target_img + str(lb)+'_'+str(tau)
		else:
			if background_attack:
				directory = 'background_'+source_img+'_'+target_img + str(lb)+'_'+str(tau)
			else:
				directory = 'foreground_'+source_img+'_'+target_img + str(lb)+'_'+str(tau)

		if not os.path.exists(directory):
			os.makedirs(directory)

		output = theano.function([self.net.x, self.net.flow],
								 [self.net.output])
		print(np.shape(x))

		grad_wrt_inp = theano.function([self.net.x, self.net.flow, self.net.y, self.net.mask, self.net.tau],
									   [theano.tensor.grad(self.net.loss, self.net.x),
										theano.tensor.grad(self.net.flow_loss, self.net.flow), self.net.loss, self.net.flow_loss, self.net.reg_term, self.net.tv_loss])
		n_vox = cfg.CONST.N_VOX
		mu = 0.85
		if target is None:
			target = np.zeros(
				(cfg.CONST.BATCH_SIZE, n_vox, 2, n_vox, n_vox))
			target[:, :, 1, :, :] = 1
			target = target.astype(theano.config.floatX)

		x_adv = x.copy()
		flow = flow_init.copy()

		y_val = target
		target = target[0, :, 1, :, :] > cfg.TEST.VOXEL_THRESH
		print('target shape: ', np.shape(target))
		results = output(x_adv, flow)
		prediction = results[0]
		predictions = prediction[0, :, 1, :, :] > cfg.TEST.VOXEL_THRESH

		boundary_mask = self.calculate_border_mask(target, weight)
		active_targets = predictions ^ target
		mask = active_targets * boundary_mask #* self.weighted_mask(active_targets)	
		mask = np.asarray([mask] * cfg.CONST.BATCH_SIZE)
		mask = np.asarray([mask] * 2)
		mask = np.transpose(mask, (1, 2, 0, 3, 4)).astype(theano.config.floatX)
		iter_ = 0
		grad_inp_t = np.zeros(x_adv.shape).astype(theano.config.floatX)
		grad_flow_t = np.zeros(flow.shape).astype(theano.config.floatX)
		#tau = 0.5
		threshold = 100

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
			#grad_x = lb*(x_adv - x) + grad_inp
			grad_x = lb*(x_adv - x)*background_mask + (lb/10)*(x_adv - x)*(1-background_mask) + grad_inp

			l2_loss = np.sum(0.5*np.multiply(x_adv-x, x_adv-x))
			print('l2 loss: ', np.sum(0.5*np.multiply(x_adv-x, x_adv-x)))

			if iter_ == 0:
				grad_inp_t = grad_x / LA.norm(grad_x)
				grad_flow_t = grad_flow / LA.norm(grad_flow)
			else:
				grad_inp_t = mu * grad_inp_t + (1 - mu) * grad_x / LA.norm(grad_x)
				grad_flow_t = mu * grad_flow_t + (1 - mu) * grad_flow / LA.norm(grad_flow)

			foreground_mask = get_foreground(x_adv)

			if dag_only:
				x_adv = np.clip(x_adv - alpha_inp*grad_inp, 0.0, 1.0)
				if not background_attack:
					x_adv = np.float32(foreground_mask * x_adv + (1 - foreground_mask) * x)
			elif spatial_only:
				flow = flow - alpha * grad_flow_t				
			else:
				if iter_ % 8 < 4:
					flow = flow - alpha * grad_flow_t
				else:
					x_adv = np.clip(x_adv - alpha_inp*grad_inp, 0.0, 1.0)
					if not background_attack:
						x_adv = np.float32(foreground_mask * x_adv + (1 - foreground_mask) * x)
			
			if iter_ % 100 == 99:				
				alpha *= 0.9
				alpha_inp *= 0.9
				threshold *= 0.6
			print("losses : ", grad[2])
			print("flow loss: ", grad[3])
			#tau = grad[2]/grad[3] 
			#print(flow)
			print('shape grad flow: ', np.shape(grad_flow))
			reg_term = grad[4]
			#print('shape reg: ', np.shape(reg_term))
			print('reg term: ', reg_term)

			iter_ += 1
			if iter_ % 200 == 199:
				#alpha *= 0.5
				#alpha_inp *= 0.5
				lb/=2
				print('lb value', lb)
				im0 = self.get_image(x_adv[0], flow[0])
				im1 = self.get_image(x_adv[1], flow[1])
				im2 = self.get_image(x_adv[2], flow[2])
				ims = np.array([im0, im1, im2])
				l2_loss = np.sqrt(np.sum(np.multiply(ims-x, ims-x)))
				save_image(ims, directory+'/flow' + str(l2_loss) + '_' + str(iter_) + '.png')
				np.save(directory+'/dag_attack' + str(iter_)+'.npy', x_adv)
				np.save(directory+'/dag_attack_flow' +  str(iter_)+'.npy', flow)
				print('savng prediction')
				voxel2obj(directory+ '/prediction' + '_targetsize:' + str(active_targets.sum()) + '_'+ str(active_targets.sum()/32768) + '_iou:' + str(iou) + '_' +
						  str(iter_) + '.obj', predictions)

			results = output(x_adv, flow)
			prediction = results[0]
			predictions = prediction[0, :, 1, :, :] > cfg.TEST.VOXEL_THRESH
			active_targets = predictions ^ target
			mask = active_targets * boundary_mask #* self.weighted_mask(active_targets)
			#mask = self.calculate_border(active_targets)
			mask = np.asarray([mask] * cfg.CONST.BATCH_SIZE)
			mask = np.asarray([mask] * 2)
			mask = np.transpose(mask, (1, 2, 0, 3, 4)).astype(theano.config.floatX)
			new_targets = active_targets.sum()

			# if new_targets - previous_targets > max(10, threshold):
			# 	print(alpha, alpha_inp)
			# 	alpha /= 2
			# 	alpha_inp /= 2
			# elif previous_targets - new_targets > max(10, threshold):
			# 	print(alpha, alpha_inp)
			# 	alpha *= 1.1
			# 	alpha_inp *= 1.1
		return prediction, None


	def test_output(self, x, y=None, mask=None, flow=None):
		'''Generate the reconstruction, loss, and activation. Evaluate loss if
		ground truth output is given. Otherwise, return reconstruction and
		activation'''
		# Cache the output function.
		if mask is None:
			mask = np.ones((cfg.CONST.BATCH_SIZE,32,2,32,32)).astype(theano.config.floatX)
			
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
			mask = np.ones((cfg.CONST.BATCH_SIZE,32,2,32,32)).astype(theano.config.floatX)
			results = self._test_output(x, y_val,mask, flow)
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
		results = self._test_output(x, y_val,mask, flow)
		print('ran')
		prediction = results[0]
		loss = results[1]
		activations = results[2:]

		if no_loss_return:
			return prediction, activations
		else:
			return prediction, loss, activations
