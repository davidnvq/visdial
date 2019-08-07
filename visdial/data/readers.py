"""
A Reader simply reads data from disk and returns it almost as is, based on
a "primary key", which for the case of VisDial v1.0 dataset, is the
``image_id``. Readers should be utilized by torch ``Dataset``s. Any type of
data pre-processing is not recommended in the reader, such as tokenizing words
to integers, embedding tokens, or passing an image through a pre-trained CNN.

Each reader must atleast implement three methods:
    - ``__len__`` to return the length of data this Reader can read.
    - ``__getitem__`` to return data based on ``image_id`` in VisDial v1.0
      dataset.
    - ``keys`` to return a list of possible ``image_id``s this Reader can
      provide data of.


Quang modified:
==============
- a bug:
```
copy.copy -> copy.deepcopy() line
dialog_for_image = copy.deepcopy(self.dialogs[image_id])

```
"""
import nltk
import copy
import json
from typing import Dict, List, Union

import h5py

# A bit slow, and just splits sentences to list of words, can be doable in
# `DialogsReader`.
from nltk.tokenize import word_tokenize
from pytorch_pretrained_bert import BertTokenizer
from tqdm import tqdm
import os
from nltk.tokenize import word_tokenize

class DialogsReader(object):
	"""
	A simple reader for VisDial v1.0 dialog data. The json file must have the
	same structure as mentioned on ``https://visualdialog.org/data``.

	Parameters
	----------
	dialogs_jsonpath : str
		Path to json file containing VisDial v1.0 train, val or test data.
	"""

	def __init__(self, config, split='train'):
		if config['model']['tokenizer'] == 'nlp':
			self.tokenize = word_tokenize
		elif config['model']['tokenizer'] == 'bert':
			self.tokenize = BertTokenizer.from_pretrained('bert-base-uncased').tokenize

		self.config = config
		path_json_dialogs = config['dataset'][split]['path_json_dialogs']

		with open(path_json_dialogs, "r") as visdial_file:
			visdial_data = json.load(visdial_file)
			self._split = visdial_data["split"]

			self.questions = visdial_data["data"]["questions"]
			self.answers = visdial_data["data"]["answers"]

			# Add empty question, answer at the end, useful for padding dialog
			# rounds for test.
			self.questions.append("")
			self.answers.append("")

			# Image_id serves as key for all three dicts here.
			self.captions = {}
			self.dialogs = {}
			self.num_rounds = {}

			for dialog_for_image in visdial_data["data"]["dialogs"]:
				self.captions[dialog_for_image["image_id"]] = dialog_for_image[
					"caption"
				]

				# Record original length of dialog, before padding.
				# 10 for train and val splits, 10 or less for test split.
				self.num_rounds[dialog_for_image["image_id"]] = len(
						dialog_for_image["dialog"]
						)

				# Pad dialog at the end with empty question and answer pairs
				# (for test split).
				while len(dialog_for_image["dialog"]) < 10:
					dialog_for_image["dialog"].append(
							{"question": -1, "answer": -1}
							)

				# Add empty answer /answer options if not provided
				# (for test split).
				for i in range(len(dialog_for_image["dialog"])):
					if "answer" not in dialog_for_image["dialog"][i]:
						dialog_for_image["dialog"][i]["answer"] = -1
					if "answer_options" not in dialog_for_image["dialog"][i]:
						dialog_for_image["dialog"][i]["answer_options"] = [-1] * 100

				self.dialogs[dialog_for_image["image_id"]] = dialog_for_image["dialog"]

			print(f"[{self._split}] Tokenizing questions...")
			for i in tqdm(range(len(self.questions))):
				self.questions[i] = self.do_tokenize(self.questions[i] + "?")

			print(f"[{self._split}] Tokenizing answers...")
			for i in tqdm(range(len(self.answers))):
				self.answers[i] = self.do_tokenize(self.answers[i])

			print(f"[{self._split}] Tokenizing captions...")
			for image_id, caption in tqdm(self.captions.items()):
				self.captions[image_id] = self.do_tokenize(caption)

		if config['model']['tokenizer'] == 'bert':
			path_feat_questions = config['dataset'][split]['path_feat_questions']
			path_feat_history = config['dataset'][split]['path_feat_history']
			self.question_reader = QuestionFeatureReader(path_feat_questions)
			self.history_reader = HistoryFeatureReader(path_feat_history)

	def __len__(self):
		return len(self.dialogs)

	def __getitem__(self, image_id: int) -> Dict[str, Union[int, str, List]]:
		caption_for_image = self.captions[image_id]
		dialog_for_image = copy.deepcopy(self.dialogs[image_id])
		num_rounds = self.num_rounds[image_id]

		# Replace question and answer indices with actual word tokens.
		dialog_for_image = self.replace_ids_by_tokens(
				dialog_for_image,
				keys=['question', 'answer', 'answer_options'])

		item = {
			'image_id'  : image_id,
			'num_rounds': num_rounds,  # 10
			"caption"   : caption_for_image,
			"dialog"    : dialog_for_image,
			}

		if self.config['model']['tokenizer'] == 'bert':
			# Replace question and answer indices with actual word tokens.
			dialog_for_image = self.replace_ids_by_tokens(
					dialog_for_image,
					keys=['answer', 'answer_options'])

			ques_feats = []
			ques_masks = []
			for i in range(len(dialog_for_image)):
				ques = self.question_reader[dialog_for_image[i]['question']]
				question_feature, question_mask = ques
				ques_feats.append(question_feature)
				ques_masks.append(question_mask)

			hist_feats = self.history_reader[image_id]

			bert_return = {
				'ques_feats': ques_feats,  # shape [10, 23, 768]
				'ques_masks': ques_masks,  # shape [10, 23]
				'hist_feats': hist_feats,  # shape [11, 768]
				'dialog'    : dialog_for_image
				}
			item.update(bert_return)

		return item

	def do_tokenize(self, text):
		tokenized_text = self.tokenize(text)
		return tokenized_text


	def replace_ids_by_tokens(self,
	                          dialog_for_image,
	                          keys=['question', 'answer', 'answer_options']):
		for dialog_round in dialog_for_image:
			for key in keys:
				if key == 'answer_options':
					for i, ans_opt in enumerate(dialog_round[key]):
						dialog_round[key][i] = self.answers[ans_opt]
				elif key == 'answer':
					dialog_round[key] = self.answers[dialog_round[key]]
				elif key == 'question':
					dialog_round[key] = self.questions[dialog_round[key]]

		return dialog_for_image


	def keys(self) -> List[int]:
		return list(self.dialogs.keys())

	@property
	def split(self):
		return self._split


def test_dialogs_reader(split='val'):
	with open('/home/ubuntu/Dropbox/repos/visdial/configs/lf_disc.json', 'r') as f:
		config = json.load(f)

	train_img_ids = [378466, 575029]
	val_img_ids = [185565, 284024]

	dialogs_reader = DialogsReader(config, split=split)
	dialog_instance = dialogs_reader[val_img_ids[0]]
	for key in dialog_instance:
		print(key)

	assert dialog_instance['image_id'] == val_img_ids[0]
	assert dialog_instance['num_rounds'] == 10

	if config['model']['encoder']['txt_embeddings']['type'] == 'lstm':
		print(dialog_instance['caption'])

	elif config['model']['encoder']['txt_embeddings']['type'] == 'bert':
		for question_mask in dialog_instance['question_masks']:
			print('question_mask', question_mask)
			print('question_mask', question_mask.dtype)
			assert question_mask.shape == (23,)

		assert dialog_instance['question_features'][0].shape == (23, 768)
		print('question_feature', dialog_instance['question_features'][0].dtype)

		print('history_feature', dialog_instance['history_feature'].shape)
		print('history_feature', dialog_instance['history_feature'].dtype)
		assert dialog_instance['history_feature'].shape == (11, 768)


class DenseAnnotationsReader(object):
	"""
	A reader for dense annotations for val split. The json file must have the
	same structure as mentioned on ``https://visualdialog.org/data``.

	Parameters
	----------
	dense_annotations_jsonpath : str
		Path to a json file containing VisDial v1.0
	"""

	def __init__(self, dense_annotations_jsonpath: str):
		if '~' in dense_annotations_jsonpath:
			dense_annotations_jsonpath = os.path.expanduser(dense_annotations_jsonpath)

		with open(dense_annotations_jsonpath, "r") as visdial_file:
			self._visdial_data = json.load(visdial_file)
			self._image_ids = [
				entry["image_id"] for entry in self._visdial_data
				]

	def __len__(self):
		return len(self._image_ids)

	def __getitem__(self, image_id: int) -> Dict[str, Union[int, List]]:
		index = self._image_ids.index(image_id)
		# keys: {"image_id", "round_id", "gt_relevance"}
		return self._visdial_data[index]

	@property
	def split(self):
		# always
		return "val"


class HistoryFeatureReader(object):
	def __init__(self, path_hdf_hist):
		self.path_hdf_hist = path_hdf_hist
		with h5py.File(path_hdf_hist, 'r') as features_hdf:
			self.image_ids = list(features_hdf['image_ids'])

	def __len__(self):
		return len(self.image_ids)

	def __getitem__(self, image_id):
		index = self.image_ids.index(image_id)
		with h5py.File(self.path_hdf_hist, 'r') as features_hdf:
			history_feature = features_hdf['features'][index]

		return history_feature


def test_history_feature_reader(hist_file='features_bert_val_history.h5', num_images=2064):
	import numpy as np

	path_hdf_hist = '/home/ubuntu/datasets/visdial/' + hist_file
	hist_reader = HistoryFeatureReader(path_hdf_hist)
	assert len(hist_reader) == num_images

	history_feature0 = hist_reader[hist_reader.image_ids[0]]
	print('history_feature shape', history_feature0.shape)
	assert history_feature0.shape == (11, 768)

	history_feature1 = hist_reader[hist_reader.image_ids[1]]
	assert id(history_feature0) != id(history_feature1)
	assert np.array_equal(history_feature0, history_feature1) == False


class QuestionFeatureReader(object):

	def __init__(self, path_hdf_ques):
		self.path_hdf_ques = path_hdf_ques
		with h5py.File(self.path_hdf_ques, 'r') as hdf:
			self.num_questions = hdf.attrs['num_questions']

	def __len__(self):
		return self.num_questions


	def __getitem__(self, question_id):
		if not os.path.isfile(self.path_hdf_ques):
			return None

		with h5py.File(self.path_hdf_ques, 'r') as hdf:
			question_feature = hdf['features'][question_id]
			question_mask = hdf['masks'][question_id]
		return question_feature, question_mask


def test_question_feature_reader():
	import numpy as np

	path_hdf_question = '/home/ubuntu/datasets/visdial/features_bert_val_questions.h5'
	ques_reader = QuestionFeatureReader(path_hdf_question)
	assert len(ques_reader) == 45237

	question_feature0, question_mask0 = ques_reader[0]
	assert question_feature0.shape == (23, 768)
	assert question_mask0.shape == (23,)

	question_feature1, question_mask1 = ques_reader[1]
	assert id(question_feature0) != id(question_feature1)
	assert np.array_equal(question_feature0, question_feature1) == False
	print('question_mask0', question_mask0)
	print('question_mask1', question_mask1)


class ImageFeaturesHdfReader(object):
	"""
	A reader for HDF files containing pre-extracted image features. A typical
	HDF file is expected to have a column named "image_id", and another column
	named "features".

	Example of an HDF file:
	```
	visdial_train_faster_rcnn_bottomup_features.h5
	   |--- "image_id" [shape: (num_images, )]
	   |--- "features" [shape: (num_images, num_proposals, feature_size)]
	   +--- .attrs ("split", "train")
	```
	Refer ``$PROJECT_ROOT/data/extract_bottomup.py`` script for more details
	about HDF structure.

	Parameters
	----------
	features_hdfpath : str
		Path to an HDF file containing VisDial v1.0 train, val or test split
		image features.
	in_memory : bool
		Whether to load the whole HDF file in memory. Beware, these files are
		sometimes tens of GBs in size. Set this to true if you have sufficient
		RAM - trade-off between speed and memory.
	"""

	def __init__(self, features_hdfpath, is_detectron=False, is_legacy=False):
		self.features_hdfpath = features_hdfpath
		self.is_legacy = is_legacy
		self.is_detectron = is_detectron

		with h5py.File(self.features_hdfpath, "r") as features_hdf:
			self._split = features_hdf.attrs["split"]
			self.image_id_list = list(features_hdf["image_id"])

	def __len__(self):
		return len(self.image_id_list)

	def __getitem__(self, image_id: int):
		index = self.image_id_list.index(image_id)
		if self.is_legacy:
			with h5py.File(self.features_hdfpath, "r") as features_hdf:
				image_id_features = features_hdf["features"][index]
			return image_id_features

		if self.is_detectron:
			with h5py.File(self.features_hdfpath, "r") as features_hdf:
				image_id_features = features_hdf["features"][index]
				boxes = features_hdf["boxes"][index]
				classes = features_hdf["classes"][index]
				scores = features_hdf["scores"][index]
			return image_id_features, boxes, classes, scores

		with h5py.File(self.features_hdfpath, "r") as features_hdf:
			image_id_features = features_hdf["features"][index]
			img_w = features_hdf["image_w"][index]
			img_h = features_hdf["image_h"][index]
			boxes = features_hdf["boxes"][index]

		return image_id_features, img_w, img_h, boxes

	def keys(self) -> List[int]:
		return self.image_id_list

	@property
	def split(self):
		return self._split
