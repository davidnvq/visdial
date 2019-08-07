from typing import Any, Dict, List, Optional
import os

import torch
from torch.nn.functional import normalize
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset
from pytorch_pretrained_bert import BertTokenizer
from visdial.data.vocabulary import Vocabulary

from visdial.data.readers import DialogsReader, DenseAnnotationsReader, ImageFeaturesHdfReader

PADDING_IDX = 0
SEP_TOKEN = 102


class VisDialDataset(Dataset):
	"""
	A full representation of VisDial v1.0 (train/val/test) dataset. According
	to the appropriate split, it returns dictionary of question, image,
	history, ground truth answer, answer options, dense annotations etc.
	"""

	def __init__(self, config, split="train", is_detectron=False):

		super().__init__()
		self.config = config
		self.split = split
		self.is_legacy = config['dataset']['is_legacy']
		self.is_detectron = is_detectron
		self.tokenizer = self._get_tokenizer(config)
		self.is_add_boundaries = self._get_is_add_boundaries(config)
		self.is_return_options = self._get_is_return_options(config, split)

		self.img_feat_reader = self._get_img_feat_reader(config, split)
		self.ann_feat_reader = self._get_ann_feat_reader(config, split)
		self.dialogs_reader = DialogsReader(config, split)
		self.image_ids = list(self.dialogs_reader.dialogs.keys())

		if config['dataset']['overfit']:
			self.image_ids = self.image_ids[:32]

	def __len__(self):
		return len(self.image_ids)

	def __getitem__(self, index, is_monitor=False):
		# Get image_id, which serves as a primary key for current instance.
		image_id = self.image_ids[index]

		visdial_instance = self.dialogs_reader[image_id]
		dialog = visdial_instance['dialog']

		if is_monitor:
			return self.monitor_output(image_id)

		item = dict()
		item['img_ids'] = torch.tensor(image_id)
		item['num_rounds'] = torch.tensor(visdial_instance['num_rounds'])

		return_elements = [
			self.return_options_to_item(dialog),
			self.return_answers_to_item(dialog),
			self.return_gt_inds_to_item(dialog),
			self.return_gt_relev_to_item(image_id),
			self.return_img_feat_to_item(image_id),
			self.return_token_feats_to_item(visdial_instance)
			]

		for elem in return_elements:
			item.update(elem)

		return item


	def _get_is_add_boundaries(self, config):
		return config['dataset']['is_add_boundaries']

	def _get_is_return_options(self, config, split):
		return config['dataset']['is_return_options']


	def _get_ann_feat_reader(self, config, split):
		path = config['dataset'][split]['path_json_dense_dialogs']
		if os.path.isfile(path):
			annotations_reader = DenseAnnotationsReader(path)
		else:
			annotations_reader = None
		return annotations_reader

	def _get_img_feat_reader(self, config, split):
		path = config['dataset'][split]['path_feat_img']
		hdf_reader = ImageFeaturesHdfReader(path, self.is_detectron, self.is_legacy)
		return hdf_reader

	def _get_tokenizer(self, config):
		if config['model']['tokenizer'] == 'bert':
			return BertTokenizer.from_pretrained("bert-base-uncased", do_lower_case=True)
		else:
			path = config['dataset']['train']['path_json_word_count']
			return Vocabulary(word_counts_path=path)


	def _pad_sequences(self, sequences: List[List[int]], max_seq_len=None):
		"""Given tokenized sequences (either questions, answers or answer
		options, tokenized in ``__getitem__``), padding them to maximum
		specified sequence length. Return as a tensor of size
		``(*, max_sequence_length)``.

		This method is only called in ``__getitem__``, chunked out separately
		for readability.

		Parameters
		----------
		sequences : List[List[int]]
			List of tokenized sequences, each sequence is typically a
			List[int].

		Returns
		-------
		torch.Tensor, torch.Tensor
			Tensor of sequences padded to max length, and length of sequences
			before padding.
		"""
		if max_seq_len is None:
			max_seq_len = self.config['dataset']['max_seq_len']

		for i in range(len(sequences)):
			if self.is_add_boundaries:
				sequences[i] = sequences[i][: max_seq_len] # + 1
			else:
				sequences[i] = sequences[i][: max_seq_len] # -1

		sequence_lengths = [len(sequence) for sequence in sequences]

		PAD_INDEX = 0

		# Pad all sequences to max_sequence_length.
		maxpadded_sequences = torch.full(
				(len(sequences), max_seq_len),
				fill_value=PAD_INDEX,
				)
		padded_sequences = pad_sequence(
				[torch.tensor(sequence) for sequence in sequences],
				batch_first=True,
				padding_value=PAD_INDEX,
				)
		maxpadded_sequences[:, : padded_sequences.size(1)] = padded_sequences
		return maxpadded_sequences.long(), torch.tensor(sequence_lengths).long()


	def _get_history(self, caption, questions, answers):
		# Allow double length of caption, equivalent to a concatenated QA pair.
		caption = caption[: self.config['dataset']['max_seq_len'] * 2]

		for i in range(len(questions)):
			questions[i] = questions[i][: self.config['dataset']['max_seq_len']]

		for i in range(len(answers)):
			if self.config['dataset']['is_add_boundaries']:
				answers[i] = answers[i][1: -1]
			else:
				answers[i] = answers[i][: self.config['dataset']['max_seq_len']]


		# History for first round is caption, else concatenated QA pair of
		# previous round.
		history = []
		history.append(caption + [self.tokenizer.EOS_INDEX])
		for question, answer in zip(questions, answers):
			history.append(question + answer + [self.tokenizer.EOS_INDEX])
		# Drop last entry from history (there's no eleventh question).
		history = history[:-1]
		max_history_length = self.config['dataset']['max_seq_len'] * 2

		round_tokens, round_lens = self.do_padding(history, max_seq_len=max_history_length)


		# 10 dialog histories
		# 1 - caption
		# 2 - caption, round1
		# 3 - caption, round1, round2
		# ....
		# 10 caption, round1, round2, ..., round9

		concat_hist_tokens = []
		for i in range(0, len(history)):
			concat_hist_tokens.append([])
			for j in range(i + 1):
				concat_hist_tokens[i].extend(history[j])

		concat_hist_tokens, concat_hist_lens = self.do_padding(
			concat_hist_tokens,
			max_seq_len=max_history_length * 10)

		return round_tokens, round_lens, concat_hist_tokens, concat_hist_lens

		# if self.split != 'test':
		# 	ten_hist_tokens = []
		# 	ten_hist_lens = []
		# 	for i in range(10):
		# 		rounds = history[0:i+1]
		# 		rounds, round_lens = self.do_padding(rounds, max_seq_len=max_history_length)
		#
		# 		ten_round_hist = torch.full((10, max_history_length), fill_value=0)
		# 		# print('rounds', rounds.size())
		# 		# print('round_lens', round_lens.size())
		# 		ten_round_hist[:rounds.size(0), :] = rounds
		#
		# 		ten_round_lens = torch.full((10,), fill_value=0)
		# 		ten_round_lens[:rounds.size(0)] = round_lens
		#
		# 		ten_hist_tokens.append(ten_round_hist)
		# 		ten_hist_lens.append(ten_round_lens)
		#
		# 	ten_hist_tokens = torch.stack(ten_hist_tokens, dim=0)
		# 	ten_hist_lens = torch.stack(ten_hist_lens, dim=0)
		# 	return ten_hist_tokens, ten_hist_lens


	def tokens_to_ids(self, tokens, is_caption=False):
		if is_caption:
			tokens = tokens[:self.config['dataset']['max_seq_len'] * 2]
		tokens = tokens[:self.config['dataset']['max_seq_len']]
		return self.tokenizer.convert_tokens_to_ids(tokens)


	def tokens_to_ids_with_boundary(self, tokens):
		tokens = tokens[:self.config['dataset']['max_seq_len'] - 2]
		tokens = [self.tokenizer.SOS_TOKEN] + tokens + [self.tokenizer.EOS_TOKEN]
		return self.tokens_to_ids(tokens)


	def convert_opt_tokens_to_ids(self, dialog):
		for round in range(len(dialog)):
			for j in range(len(dialog[round]["answer_options"])):
				tokens = dialog[round]["answer_options"][j]
				if self.is_add_boundaries:
					dialog[round]["answer_options"][j] = self.tokens_to_ids_with_boundary(tokens)
				else:
					dialog[round]["answer_options"][j] = self.tokens_to_ids(tokens)
		return dialog

	def do_padding(self, sequences, start=None, end=None, max_seq_len=None):
		sequences = [seq[start:end] for seq in sequences]
		sequences, seq_lens = self._pad_sequences(sequences, max_seq_len)
		return sequences, seq_lens

	def return_options_to_item(self, dialog):
		self.convert_opt_tokens_to_ids(dialog)
		ans_opts_in, ans_opts_out = [], []
		ans_opts_in_len = []
		ans_opts_out_len = []
		ans_opts_len = []
		ans_opts = []

		for dialog_round in dialog:
			# for boundary
			# answer options input
			result = self.do_padding(dialog_round['answer_options'], end=-1)
			ans_opts_in.append(result[0])
			ans_opts_in_len.append(result[1])

			# answer options output
			result = self.do_padding(dialog_round['answer_options'], start=1)
			ans_opts_out.append(result[0])
			ans_opts_out_len.append(result[1])

			# for normal case
			result = self.do_padding(dialog_round['answer_options'], start=1, end=-1)
			ans_opts.append(result[0])
			ans_opts_len.append(result[1])

		ans_opts = torch.stack(ans_opts, dim=0)
		ans_opts_in = torch.stack(ans_opts_in, dim=0)
		ans_opts_out = torch.stack(ans_opts_out, dim=0)
		return {
			'opts'    : ans_opts,
			'opts_in' : ans_opts_in,
			'opts_out': ans_opts_out,
			'opts_len': torch.stack(ans_opts_len, dim=0),
			'opts_in_len': torch.stack(ans_opts_in_len, dim=0),
			'opts_out_len': torch.stack(ans_opts_out_len, dim=0)
			}

	def return_answers_to_item(self, dialog):
		for round in range(len(dialog)):
			tokens = [self.tokenizer.SOS_TOKEN] + dialog[round]['answer'] + [self.tokenizer.EOS_TOKEN]
			dialog[round]['answer'] = self.tokens_to_ids(tokens)

		round_answers = [dialog_round["answer"] for dialog_round in dialog]
		result = self.do_padding(round_answers, start=1, end=-1)
		result_in = self.do_padding(round_answers, end=-1)
		result_out = self.do_padding(round_answers, start=1)

		return {
			'ans'    : result[0],
			'ans_in' : result_in[0],
			'ans_out': result_out[0],
			'ans_len': result[1],
			'ans_in_len': result_in[1],
			'ans_out_len': result_out[1],
			}

	def return_gt_inds_to_item(self, dialog):
		if 'test' not in self.split:
			answer_indices = [dialog_round['gt_index'] for dialog_round in dialog]
			return {'ans_ind': torch.tensor(answer_indices).long()}
		else:
			return {}

	def return_gt_relev_to_item(self, image_id):
		if "val" in self.split:
			dense_annotations = self.ann_feat_reader[image_id]
			return {
				"gt_relevance": torch.tensor(dense_annotations["gt_relevance"]).float(),
				"round_id"    : torch.tensor(dense_annotations["round_id"]).long()
				}
		else:
			return {}

	def return_bert_feats_to_item(self, visdial_instance):
		# get bert features
		if self.config['model']['encoder']['txt_embeddings']['type'] == 'bert':
			return {
				'ques_feats': torch.tensor(visdial_instance['ques_feats']),
				'hist_feats': torch.tensor(visdial_instance['hist_feats']),
				'ques_masks': torch.tensor(visdial_instance['ques_masks']).long()
				}
		else:
			return {}

	def return_token_feats_to_item(self, visdial_instance):
		if self.config['model']['tokenizer'] == 'nlp':
			caption = visdial_instance["caption"]
			dialog = visdial_instance["dialog"]

			# Convert word tokens of caption, question
			caption = self.tokens_to_ids(caption)

			for i in range(len(dialog)):
				dialog[i]["question"] = self.tokens_to_ids(dialog[i]["question"])

			sequences = [dialog_round["question"] for dialog_round in dialog]
			ques_tokens, ques_lens = self._pad_sequences(sequences)

			hist_tokens, hist_lens, concat_hist_tokens, concat_hist_lens = self._get_history(
					caption,
					[dialog_round["question"] for dialog_round in dialog],
					[dialog_round["answer"] for dialog_round in dialog],
					)

			return {
				'ques_tokens': ques_tokens.long(),
				'hist_tokens': hist_tokens.long(),
				'ques_len'   : ques_lens.long(),
				'hist_len'   : hist_lens.long(),
				'concat_hist_tokens' : concat_hist_tokens.long(),
				'concat_hist_len' : concat_hist_lens.long()
				}
		else:
			return {}

	def return_img_feat_to_item(self, image_id):
		# Get image features for this image_id using hdf reader.
		if self.is_legacy:
			img_feat = self.img_feat_reader[image_id]
			img_feat = torch.tensor(img_feat)
			if self.config['dataset']["img_norm"]:
				img_feat = normalize(img_feat, dim=0, p=2)

			return {'img_feat' : img_feat}

		if self.is_detectron:
			img_feat, boxes, classes, scores = self.img_feat_reader[image_id]
			img_feat = torch.tensor(img_feat)
			boxes = torch.tensor(boxes)

			# Normalize image features at zero-th dimension
			# (since there's no batch dimension).
			if self.config['dataset']["img_norm"]:
				img_feat = normalize(img_feat, dim=0, p=2)

			return {'img_feat': img_feat,
			        'boxes'   : boxes}

		else:
			img_feat, img_w, img_h, boxes  = self.img_feat_reader[image_id]
			img_feat = torch.tensor(img_feat)
			img_w = torch.tensor(img_w)
			img_h = torch.tensor(img_h)
			boxes = torch.tensor(boxes)

			# Normalize image features at zero-th dimension
			# (since there's no batch dimension).
			if self.config['dataset']["img_norm"]:
				img_feat = normalize(img_feat, dim=0, p=2)

			return {'img_feat': img_feat,
			        'img_w' : img_w,
			        'img_h' : img_h,
			        'boxes' : boxes}

	def monitor_output(self, image_id):
		visdial_instance = self.dialogs_reader[image_id]
		dialog = visdial_instance['dialog']

		if "val" in self.split:
			dense_annotations = self.ann_feat_reader[image_id]

		gt_relevance = torch.tensor(dense_annotations["gt_relevance"]).float()
		round_id = torch.tensor(dense_annotations["round_id"]).long()
		rel_ans_idx = dialog[round_id - 1]["gt_index"]
		caption = 'caption should be extracted'
		return {
			'img_id'      : image_id,
			'caption'     : caption,
			'dialog'      : dialog,
			'gt_relevance': gt_relevance,
			'round_id'    : round_id,
			'rel_ans_idx' : rel_ans_idx
			}


def get_config():
	import json

	path = os.path.expanduser('~/Dropbox/repos/visdial/configs/lf_bert_disc.json')
	with open(path, 'r') as f:
		config = json.load(f)
	return config


def test_visdial_dataset(config):
	dataset = VisDialDataset(config, split='val')
	item = dataset[0]
	for key in item:
		print("{:<20}".format(key), item[key].shape)

	dataloader = torch.utils.data.DataLoader(dataset, batch_size=1)
	for batch in dataloader:
		break
	return dataset, dataloader, batch
