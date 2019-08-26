from pathlib import Path
import warnings
import os

import torch
from torch import nn, optim
import yaml


class CheckpointManager(object):
	"""A checkpoint manager saves state dicts of model and optimizer
	as .pth files in a specified directory. This class closely follows
	the API of PyTorch optimizers and learning rate schedulers.

	Note::
		For ``DataParallel`` modules, ``model.module.state_dict()`` is
		saved, instead of ``model.state_dict()``.

	Parameters
	----------
	model: nn.Module
		Wrapped model, which needs to be checkpointed.
	optimizer: optim.Optimizer
		Wrapped optimizer which needs to be checkpointed.
	checkpoint_dirpath: str
		Path to an empty or non-existent directory to save checkpoints.

	Example
	--------
	>>> model = torch.nn.Linear(10, 2)
	>>> optimizer = torch.optim.Adam(model.parameters())
	>>> ckpt_manager = CheckpointManager(model, optimizer, "/tmp/ckpt")
	>>> for epoch in range(20):
	...     for batch in dataloader:
	...         do_iteration(batch)
	...     ckpt_manager.step()
	"""

	def __init__(
			self,
			model,
			optimizer,
			checkpoint_dirpath,
			config={}
			):

		if not isinstance(model, nn.Module):
			raise TypeError("{} is not a Module".format(type(model).__name__))

		if not isinstance(optimizer, optim.Optimizer):
			raise TypeError(
					"{} is not an Optimizer".format(type(optimizer).__name__)
					)

		self.model = model
		self.optimizer = optimizer
		self.ckpt_dirpath = Path(checkpoint_dirpath)
		self.best_ndcg = 0.0
		self.best_mean = 100.
		self.init_directory(config)
		self.best_ndcg_epoch = 0
		self.best_mean_epoch = 0

	def init_directory(self, config={}):
		"""init"""
		self.ckpt_dirpath.mkdir(parents=True, exist_ok=True)

		import json
		with open(self.ckpt_dirpath/'config.json', 'w') as f:
			json.dump(config, f)


	def step(self, epoch=None, only_best=False, metrics=None, key=''):
		"""Save checkpoint if step size conditions meet. """
		if not only_best:
			self._save_state_dict(str(epoch), epoch, metrics)

			if metrics[key + 'ndcg'] >= self.best_ndcg:
				self.best_ndcg = metrics[key + 'ndcg']
				self.best_ndcg_epoch = epoch

			if metrics[key + 'mean'] >= self.best_ndcg:
				self.best_mean = metrics[key + 'mean']
				self.best_mean_epoch = epoch

		else:
			if metrics[key + 'ndcg'] >= self.best_ndcg:
				self.best_ndcg = metrics[key + 'ndcg']
				self.best_ndcg_epoch = epoch
				print('Save best ndcg {} at epoch {}'.format(self.best_ndcg, epoch))
				self._save_state_dict('best_ndcg', epoch, metrics)

			if metrics[key + 'mean'] <= self.best_mean:
				self.best_mean = metrics[key + 'mean']
				self.best_ndcg_epoch = epoch
				print('Save best mean {} at epoch {}'.format(self.best_mean, epoch))
				self._save_state_dict('best_mean', epoch, metrics)

		self._save_state_dict('last', epoch, metrics)


	def _save_state_dict(self, name, epoch, metrics):
		"""save state_dict"""
		state_dict = {'model'    : self._model_state_dict(),
		              'optimizer': self.optimizer.state_dict(),
		              'epoch'    : epoch,
		              'metrics'  : metrics}
		ckpt_path = self.ckpt_dirpath / f"checkpoint_{name}.pth"
		torch.save(state_dict, ckpt_path)

	def _model_state_dict(self):
		"""Returns state dict of model, taking care of DataParallel case."""
		if isinstance(self.model, nn.DataParallel):
			return self.model.module.state_dict()
		else:
			return self.model.state_dict()


def load_checkpoint(model, optimizer, checkpoint_pthpath=None, device='cuda', resume=False):
	"""Load checkpoint"""
	# load encoder, decoder, optimizer state_dicts

	if checkpoint_pthpath is not None:
		components = torch.load(checkpoint_pthpath, map_location=device)
		print("Loaded model from {}".format(checkpoint_pthpath))
		print('At epoch:', components.get('epoch'))
		print('Metrics score:', components.get('metrics'))
	else:
		print("Can't load weight from {}".format(checkpoint_pthpath))
		return 0, model, optimizer

	if resume:
		# "path/to/checkpoint_xx.pth" -> xx
		print('Resume training....')
		start_epoch = components['epoch']
		model.load_my_state_dict(components["model"])
		optimizer.load_state_dict(components["optimizer"])
		return start_epoch, model, optimizer

	else:
		model.load_my_state_dict(components["model"])
		return 0, model, optimizer

def load_checkpoint_from_config(model, optimizer, config):

	return load_checkpoint(model, optimizer,
						   checkpoint_pthpath=config['callbacks']['path_pretrained_ckpt'],
						   resume=config['callbacks']['resume'],
						   device='cuda')
