# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Extract pre-computed feature vectors from a PyTorch BERT model."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tqdm import tqdm
import h5py
import argparse
import collections
import logging
import json
import re

import torch
from torch.utils.data import TensorDataset, DataLoader, SequentialSampler
from torch.utils.data.distributed import DistributedSampler

from pytorch_pretrained_bert.tokenization import BertTokenizer
from pytorch_pretrained_bert.modeling import BertModel

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)


class InputExample(object):

	def __init__(self, image_id, unique_id, text_a, text_b):
		self.image_id = image_id
		self.unique_id = unique_id
		self.text_a = text_a
		self.text_b = text_b


class InputFeatures(object):
	"""A single set of features of data."""

	def __init__(self, image_id, unique_id, tokens, input_ids, input_mask, input_type_ids):
		self.image_id = image_id
		self.unique_id = unique_id
		self.tokens = tokens
		self.input_ids = input_ids
		self.input_mask = input_mask
		self.input_type_ids = input_type_ids


def convert_examples_to_features(examples, seq_length, tokenizer):
	"""Loads a data file into a list of `InputFeature`s."""

	features = []
	for (ex_index, example) in enumerate(examples):
		tokens_a = tokenizer.tokenize(example.text_a)

		tokens_b = None
		if example.text_b:
			tokens_b = tokenizer.tokenize(example.text_b)

		if tokens_b:
			# Modifies `tokens_a` and `tokens_b` in place so that the total
			# length is less than the specified length.
			# Account for [CLS], ?, [SEP], [SEP] with "- 3"
			_truncate_seq_pair(tokens_a, tokens_b, seq_length - 4)
		else:
			# Account for [CLS] and ? and [SEP] with "- 3"
			if len(tokens_a) > seq_length - 3:
				tokens_a = tokens_a[0:(seq_length - 3)]

		# The convention in BERT is:
		# (a) For sequence pairs:
		#  tokens:   [CLS] is this jack ##son ##ville ? [SEP] no it is not . [SEP]
		#  type_ids:   0   0  0    0    0     0      0   0    1  1  1   1  1   1
		# (b) For single sequences:
		#  tokens:   [CLS] the dog is hairy . [SEP]
		#  type_ids:   0   0   0   0  0     0   0
		#
		# Where "type_ids" are used to indicate whether this is the first
		# sequence or the second sequence. The embedding vectors for `type=0` and
		# `type=1` were learned during pre-training and are added to the wordpiece
		# embedding vector (and position vector). This is not *strictly* necessary
		# since the [SEP] token unambigiously separates the sequences, but it makes
		# it easier for the model to learn the concept of sequences.
		#
		# For classification tasks, the first vector (corresponding to [CLS]) is
		# used as as the "sentence vector". Note that this only makes sense because
		# the entire model is fine-tuned.
		tokens = []
		input_type_ids = []
		tokens.append("[CLS]")
		input_type_ids.append(0)
		for token in tokens_a:
			tokens.append(token)
			input_type_ids.append(0)

		if example.unique_id > 0:
			tokens.append("?")
			input_type_ids.append(0)

		tokens.append("[SEP]")
		input_type_ids.append(0)

		if tokens_b:
			for token in tokens_b:
				tokens.append(token)
				input_type_ids.append(1)
			tokens.append("[SEP]")
			input_type_ids.append(1)

		input_ids = tokenizer.convert_tokens_to_ids(tokens)

		# The mask has 1 for real tokens and 0 for padding tokens. Only real
		# tokens are attended to.
		input_mask = [1] * len(input_ids)

		# Zero-pad up to the sequence length.
		while len(input_ids) < seq_length:
			input_ids.append(0)
			input_mask.append(0)
			input_type_ids.append(0)

		assert len(input_ids) == seq_length
		assert len(input_mask) == seq_length
		assert len(input_type_ids) == seq_length

		if ex_index < 20:
			logger.info("*** Example ***")
			logger.info("unique_id: %s" % (example.unique_id))
			logger.info("{: <20}".format("tokens:") + "%s" % " ".join([str(x) for x in tokens]))
			logger.info("{: <20}".format("input_ids:") +  "%s" % " ".join([str(x) for x in input_ids]))
			logger.info("{: <20}".format("input_mask:") + "%s" % " ".join([str(x) for x in input_mask]))
			logger.info(
					"{: <20}".format("input_type_ids:") + "%s" % " ".join([str(x) for x in input_type_ids]))

		features.append(
				InputFeatures(
						image_id=example.image_id,
						unique_id=example.unique_id,
						tokens=tokens,
						input_ids=input_ids,
						input_mask=input_mask,
						input_type_ids=input_type_ids))
	return features


def _truncate_seq_pair(tokens_a, tokens_b, max_length):
	"""Truncates a sequence pair in place to the maximum length."""

	# This is a simple heuristic which will always truncate the longer sequence
	# one token at a time. This makes more sense than truncating an equal percent
	# of tokens from each, since if one sequence is very short then each token
	# that's truncated likely contains more information than a longer sequence.
	while True:
		total_length = len(tokens_a) + len(tokens_b)
		if total_length <= max_length:
			break
		if len(tokens_a) > len(tokens_b):
			tokens_a.pop()
		else:
			tokens_b.pop()


def read_history_examples(path_json_dialogs):
	with open(path_json_dialogs, "r") as visdial_file:
		visdial_data = json.load(visdial_file)
		questions = visdial_data["data"]["questions"]
		answers = visdial_data['data']['answers']

	examples = []
	for dialog_for_image in visdial_data['data']['dialogs']:
		image_id = dialog_for_image['image_id']
		caption = dialog_for_image['caption']
		num_rounds = len(dialog_for_image['dialog']) + 1  # include caption
		assert num_rounds == 11

		for i in range(num_rounds):
			if i == 0:
				examples.append(
						InputExample(image_id=image_id, unique_id=i, text_a=caption, text_b=None)
						)
			else:
				examples.append(
						InputExample(
								image_id=image_id, unique_id=i,
								text_a=questions[dialog_for_image['dialog'][i - 1]['question']],
								text_b=answers[dialog_for_image['dialog'][i - 1]['answer']]))
	return examples


def main():
	parser = argparse.ArgumentParser()

	## Required parameters
	parser.add_argument("--input_file", default='/home/ubuntu/datasets/visdial/visdial_1.0_train.json')
	parser.add_argument("--output_file", default='/home/ubuntu/datasets/visdial/features_bert_train_history.h5')
	parser.add_argument("--token-file", default='/home/ubuntu/datasets/visdial/tokens_bert_train_history.json')
	parser.add_argument("--bert_model", default='bert-base-uncased', type=str,
	                    help="Bert pre-trained model selected in the list: "
	                         "[bert-base-uncased, bert-large-uncased, bert-base-cased, "
	                         "bert-base-multilingual, bert-base-chinese].")

	## Other parameters
	parser.add_argument("--do_lower_case", action='store_true', help="Set this flag if you are using an uncased model.")
	parser.add_argument("--layers", default="-1,-2,-3,-4", type=str)
	parser.add_argument("--max_seq_length", default=40 + 3, type=int,
	                    help="The maximum total input sequence length after WordPiece tokenization. Sequences longer "
	                         "than this will be truncated, and sequences shorter than this will be padded.")
	parser.add_argument("--batch_size", default=11, type=int, help="Batch size for predictions.")
	parser.add_argument("--device", default="cuda:0")
	parser.add_argument("--num-rounds", default=11, type=int)
	parser.add_argument("--feat-dim", default=768, type=int)

	args = parser.parse_args()

	device = torch.device(args.device)

	layer_indexes = [int(x) for x in args.layers.split(",")]

	tokenizer = BertTokenizer.from_pretrained(args.bert_model, do_lower_case=args.do_lower_case)

	examples = read_history_examples(args.input_file)

	num_images = len(examples) / 11
	
	print('num_examples =', num_images)

	features = convert_examples_to_features(
			examples=examples, seq_length=args.max_seq_length, tokenizer=tokenizer)

	unique_id_to_feature = {}
	for feature in features:
		unique_id_to_feature[feature.unique_id] = feature

	model = BertModel.from_pretrained(args.bert_model)
	model.to(device)

	all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
	all_input_mask = torch.tensor([f.input_mask for f in features], dtype=torch.long)
	all_input_type_ids = torch.tensor([f.input_type_ids for f in features], dtype=torch.long)
	all_example_index = torch.arange(all_input_ids.size(0), dtype=torch.long)

	eval_data = TensorDataset(all_input_ids, all_input_mask, all_input_type_ids, all_example_index)
	eval_sampler = SequentialSampler(eval_data)
	eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=args.batch_size)

	model.eval()

	history_tokens_dict = {}
	with h5py.File(args.output_file, 'w') as writer:
		features_h5d = writer.create_dataset('features',
		                                     (num_images, args.num_rounds, args.feat_dim))
		image_ids_h5d = writer.create_dataset("image_ids", (num_images,), dtype=int)

		for idx, (input_ids, input_mask, input_type_ids, example_indices) in enumerate(tqdm(eval_dataloader)):
			input_ids = input_ids.to(device)
			input_mask = input_mask.to(device)
			input_type_ids = input_type_ids.to(device)

			all_encoder_layers, _ = model(input_ids,
			                              token_type_ids=input_type_ids,
			                              attention_mask=input_mask)

			all_encoder_layers = all_encoder_layers

			init_image_id = int(features[example_indices[0]].image_id)
			# print('image_id', init_image_id)
			init_unique_id = int(features[example_indices[0]].unique_id)

			history_tokens_dict[init_image_id] = []
			history_feature = []

			assert len(example_indices) == 11

			for b, example_index in enumerate(example_indices):
				feature = features[example_index.item()]
				image_id = int(feature.image_id)
				unique_id = int(feature.unique_id)
				assert unique_id == init_unique_id + b
				assert image_id == init_image_id

				history_tokens_dict[init_image_id].append(feature.tokens)

				example_feature = None
				for (j, layer_index) in enumerate(layer_indexes):
					if example_feature is None:
						# [CLS] feature - The first token
						example_feature = all_encoder_layers[int(layer_index)][b][0:1].detach().cpu()
					else:
						# [CLS] feature - The first token
						example_feature += all_encoder_layers[int(layer_index)][b][0:1].detach().cpu()

				assert example_feature.shape == (1, args.feat_dim)
				history_feature.append(example_feature)

			history_feature = torch.cat(history_feature, dim=0)
			assert history_feature.shape == (args.num_rounds, args.feat_dim)

			features_h5d[idx] = history_feature
			image_ids_h5d[idx] = init_image_id

	with open(args.token_file, 'w', encoding='utf-8') as f:
		json.dump(history_tokens_dict, f)


if __name__ == "__main__":
	main()
