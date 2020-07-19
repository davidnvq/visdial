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

Credit:
https://github.com/batra-mlp-lab/visdial-challenge-starter-pytorch/blob/master/visdialch/data/readers.py
With some modification in For ImageReader
"""
import nltk
import copy
import json
from typing import Dict, List, Union

import h5py

# A bit slow, and just splits sentences to list of words, can be doable in
# `DialogsReader`.
from nltk.tokenize import word_tokenize
# from pytorch_pretrained_bert import BertTokenizer
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
        if config['model']['txt_tokenizer'] == 'nlp':
            self.tokenize = word_tokenize
        elif config['model']['txt_tokenizer'] == 'bert':
            pass
        # self.tokenize = BertTokenizer.from_pretrained('bert-base-uncased').tokenize

        self.config = config

        path_json_dialogs = config['dataset'][f"{split}_json_dialog_path"]
        path_json_dialogs = os.path.expanduser(path_json_dialogs)

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
            for i in range(len(self.questions)):
                # print('len(self.questions[i])', len(self.questions[i]))
                self.questions[i] = self.questions[i] + "?" if len(self.questions[i]) > 0 else self.questions[i]
                self.questions[i] = self.do_tokenize(self.questions[i])

            print(f"[{self._split}] Tokenizing answers...")
            for i in range(len(self.answers)):
                self.answers[i] = self.do_tokenize(self.answers[i])

            print(f"[{self._split}] Tokenizing captions...")
            for image_id, caption in self.captions.items():
                self.captions[image_id] = self.do_tokenize(caption)

        if config['model']['txt_tokenizer'] == 'bert':
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
            'image_id': image_id,
            'num_rounds': num_rounds,  # 10
            "caption": caption_for_image,
            "dialog": dialog_for_image,
        }

        if self.config['model']['txt_tokenizer'] == 'bert':
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
                'dialog': dialog_for_image
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


class ImageFeaturesHdfReader:
    """
    Parameters
    ----------
    features_hdfpath : str
        Path to an HDF file containing VisDial v1.0 train, val or test split
        image features.
    in_memory : bool
        Whether to load the whole HDF file in memory. Beware, these files are
        sometimes tens of GBs in size. Set this to true if you have sufficient
        RAM - trade-off between speed and memory.

    ['boxes',
     'cls_indices',
     'features',
     'image_h',
     'image_id',
     'image_w',
     'num_boxes',
     'st1_boxes',
     'top_attrs',
     'top_attrs_scores']
    """

    def __init__(self,
                 hdf_path,
                 genome_path='datasets/genome/1600-400-20'):

        self.hdf_path = hdf_path
        genome_path = os.path.expanduser(genome_path)
        self.genome_path = genome_path
        with h5py.File(self.hdf_path, "r") as hdf:
            self._split = hdf.attrs["split"]
            self.image_id_list = list(hdf["image_id"])
            self.dataset_names = list(hdf.keys())

        # Load classes
        self.CLASSES = ['__background__']
        with open(os.path.join(genome_path, 'objects_vocab.txt')) as f:
            for object in f.readlines():
                self.CLASSES.append(object.split(',')[0].lower().strip())

        # Load attributes
        self.ATTRIBUTES = ['__no_attribute__']
        with open(os.path.join(genome_path, 'attributes_vocab.txt')) as f:
            for att in f.readlines():
                self.ATTRIBUTES.append(att.split(',')[0].lower().strip())

    def __len__(self):
        return len(self.image_id_list)

    def __getitem__(self, image_id: int):
        index = self.image_id_list.index(image_id)

        output = {}

        with h5py.File(self.hdf_path, 'r') as hdf:
            for name in self.dataset_names:
                if name not in ['st1_boxes']:
                    output[name] = hdf[name][index]

        if 'cls_indices' in self.dataset_names:
            output['cls_names'] = self.get_cls_name(output['cls_indices'])
        if 'top_attrs' in self.dataset_names:
            output['top_attr_names'] = self.get_attr_names(output['top_attrs'])
        return output

    def get_cls_name(self, cls_indices):
        cls_names = []
        for cls_idx in cls_indices:
            cls_names.append(self.CLASSES[cls_idx])
        return cls_names

    def get_attr_names(self, top_attrs):
        top_word_attrs = []
        for row_attrs in top_attrs:
            box_word_attrs = []
            for attr_idx in row_attrs:
                box_word_attrs.append(self.ATTRIBUTES[attr_idx])
            top_word_attrs.append(box_word_attrs)
        return top_word_attrs

    def keys(self) -> List[int]:
        return self.image_id_list

    @property
    def split(self):
        return self._split
