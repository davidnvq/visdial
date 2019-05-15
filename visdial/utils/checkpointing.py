"""
A checkpoint manager periodically saves model and optimizer as .pth
files during training.

Checkpoint managers help with experiment reproducibility, they record
the commit SHA of your current codebase in the checkpoint saving
directory. While loading any checkpoint from other commit, they raise a
friendly warning, a signal to inspect commit diffs for potential bugs.
Moreover, they copy experiment hyper-parameters as a YAML config in
this directory.

That said, always run your experiments after committing your changes,
this doesn't account for untracked or staged, but uncommitted changes.
"""
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
			**kwargs,
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
		self.init_directory(**kwargs)

	def init_directory(self, config={}):
		"""init"""
		self.ckpt_dirpath.mkdir(parents=True, exist_ok=True)
		with open(str(self.ckpt_dirpath / "config.yml"), "w") as file:
			yaml.dump(config, file, default_flow_style=False)

	def step(self, epoch=None, only_best=False, metrics=None):
		"""Save checkpoint if step size conditions meet. """
		if not only_best:
			self._save_state_dict(str(epoch), epoch, metrics)
		else:
			if metrics['ndcg'] >= self.best_ndcg:
				self.best_ndcg = metrics['ndcg']
				self._save_state_dict('best_ndcg', epoch, metrics)

			if metrics['mean'] <= self.best_mean:
				self.best_mean = metrics['mean']
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


def update_weights(net, pretrained_dict):
	model_dict = net.state_dict()
	pretrained_dict = {k: v for k, v in pretrained_dict.items()
	                   if k in model_dict}

	# for old lf_disc on unidirectional-LSTM
	incompat_keys = [
		'encoder.hist_rnn.rnn_model.weight_ih_l1',
		'encoder.ques_rnn.rnn_model.weight_ih_l1',
		'decoder.option_rnn.rnn_model.weight_ih_l1'
		]

	for key in incompat_keys:
		pretrained_dict[key] = torch.cat([pretrained_dict[key]] * 2, dim=-1)

	model_dict.update(pretrained_dict)
	net.load_state_dict(model_dict)
	return net


def load_checkpoint(checkpoint_pthpath, model, optimizer=None, device='cuda', resume=False):
	"""Load checkpoint"""
	# load encoder, decoder, optimizer state_dicts

	if os.path.exists(checkpoint_pthpath) and os.path.isfile(checkpoint_pthpath):
		components = torch.load(checkpoint_pthpath, map_location=device)
		print("Loaded model from {}".format(checkpoint_pthpath))
	else:
		print("Can't load weight from {}".format(checkpoint_pthpath))
		return model

	if resume:
		# "path/to/checkpoint_xx.pth" -> xx
		start_epoch = int(checkpoint_pthpath.split("_")[-1][:-4]) + 1
		model.load_state_dict(components["model"])
		optimizer.load_state_dict(components["optimizer"])
		return start_epoch, model, optimizer

	else:
		return update_weights(model, components["model"])
