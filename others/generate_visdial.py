# This code can be copy-paste to the Jupyter Notebook in the same folder to run.
# set up Python environment: numpy for numerical routines, and matplotlib for plotting

import os
import sys
import cv2
import h5py
import csv
import pylab
import base64
import argparse
import numpy as np
from glob import glob
from tqdm import tqdm
from skimage import transform
import matplotlib.pyplot as plt

# set display defaults
plt.rcParams['figure.figsize'] = (20, 12)  # small images
plt.rcParams['image.interpolation'] = 'nearest'  # don't interpolate: show square pixels
plt.rcParams['image.cmap'] = 'gray'  # use grayscale output rather than a (potentially misleading) color heatmap

# Change dir to caffe root or prototxt database paths won't work wrong
os.chdir('..')
os.getcwd()

FIRST_STAGE = False
conf_thresh = 0.2
attr_thresh = 0.1

NUM_BOXES = [(100, 100)]  # The min boxes and the max boxes.
FIELDNAMES = ['image_id', 'image_h', 'image_w', 'num_boxes', 'top_attrs', 'top_attrs_scores', 'boxes', 'features',
              'st1_boxes']

# The caffe module needs to be on the Python path;
#  we'll add it here explicitly.
sys.path.insert(0, './caffe/python/')
sys.path.insert(0, './lib/')
sys.path.insert(0, './tools/')

import caffe
# Check object extraction
from fast_rcnn.config import cfg, cfg_from_file
from fast_rcnn.test import im_detect, _get_blobs
from fast_rcnn.nms_wrapper import nms

data_path = './data/genome/1600-400-20'
cfg_from_file('experiments/cfgs/faster_rcnn_end2end_resnet.yml')

# Load classes
CLASSES = ['__background__']
with open(os.path.join(data_path, 'objects_vocab.txt')) as f:
    for object in f.readlines():
        CLASSES.append(object.split(',')[0].lower().strip())

# Load attributes
ATTRIBUTES = ['__no_attribute__']
with open(os.path.join(data_path, 'attributes_vocab.txt')) as f:
    for att in f.readlines():
        ATTRIBUTES.append(att.split(',')[0].lower().strip())

GPU_ID = 0  # if we have multiple GPUs, pick one
caffe.set_device(GPU_ID)
caffe.set_mode_gpu()
net = None


def get_nms_boxes(st2_scores, st1_boxes):
    max_cls_scores = np.zeros((st1_boxes.shape[0]))
    max_cls_indices = np.zeros((st1_boxes.shape[0]), dtype=int)

    # Keep only the best class_box for each box (each row in st2_boxes has 1601 boxes for 1 ROI)
    for cls_ind in range(1, st2_scores.shape[1]):
        cls_ind_scores = st2_scores[:, cls_ind]
        dets = np.hstack((st1_boxes, cls_ind_scores[:, np.newaxis])).astype(np.float32)
        keep = np.array(nms(dets, cfg.TEST.NMS))
        max_cls_indices[keep] = np.where(cls_ind_scores[keep] > max_cls_scores[keep],
                                         cls_ind, max_cls_indices[keep])
        max_cls_scores[keep] = np.where(cls_ind_scores[keep] > max_cls_scores[keep],
                                        cls_ind_scores[keep], max_cls_scores[keep])

    return max_cls_scores, max_cls_indices


def extract_im(net, img_path):
    img = cv2.imread(img_path)

    st2_scores, st2_boxes, st2_attr_scores, st2_rel_scores = im_detect(net, img)
    pool5 = net.blobs['pool5_flat'].data

    # unscale back to raw image space
    blobs, im_scales = _get_blobs(img, None)

    # Keep the original boxes, don't worry about the regression bbox outputs
    rois = net.blobs['rois'].data.copy()
    st1_scores = rois[:, 0]
    st1_boxes = rois[:, 1:5] / im_scales[0]

    # Keep only the best class_box of each row in st2_boxes has 1601 boxes for 1 ROI
    max_cls_scores, max_cls_indices = get_nms_boxes(st2_scores, st1_boxes)

    # For each threshold of boxes,
    # save (keep_box_indices, keep_box_cls_indices)
    keep_ind = []
    for (min_boxes, max_boxes) in NUM_BOXES:
        keep_box_indices = np.where(max_cls_scores >= conf_thresh)[0]

        if len(keep_box_indices) < min_boxes:
            keep_box_indices = np.argsort(max_cls_scores)[::-1][:min_boxes]
        elif len(keep_box_indices) > max_boxes:
            keep_box_indices = np.argsort(max_cls_scores)[::-1][:max_boxes]

        # print("keep_box_indices len", len(keep_box_indices))
        keep_box_cls_indices = max_cls_indices[keep_box_indices]
        keep_ind.append((keep_box_indices, keep_box_cls_indices))

    return {
        "image_id": image_id_from_path(img_path),
        "image_h": np.size(img, 0),
        "image_w": np.size(img, 1),
        "keep_ind": keep_ind,
        "st2_scores": st2_scores,
        "st2_boxes": st2_boxes,
        "st2_attr_scores": st2_attr_scores,
        "pool5": pool5,
        "st1_boxes": st1_boxes
    }


def get_topN_attrs(st2_attr_scores, keep_box_indices, topN=20):
    attrs = st2_attr_scores[keep_box_indices]

    # shape [num_boxes, topN]
    top_attrs = np.zeros((attrs.shape[0], topN), dtype=int)
    top_attrs_scores = np.zeros((attrs.shape[0], topN))

    for i, box_attr in enumerate(attrs):
        top_attr = np.argsort(box_attr)[::-1][:topN]
        top_attr_score = box_attr[top_attr]
        top_attrs[i] = top_attr
        top_attrs_scores[i] = top_attr_score
    return top_attrs, top_attrs_scores


def get_topN_attrs(st2_attr_scores, keep_box_indices, topN=20, attr_thresh=0.1):
    attrs = st2_attr_scores[keep_box_indices]

    # shape [num_boxes, topN]
    top_attrs = np.zeros((attrs.shape[0], topN), dtype=int)
    top_attrs_scores = np.zeros((attrs.shape[0], topN))

    for i, box_attr in enumerate(attrs):
        # except __no_attribute__
        top_attr = np.argsort(box_attr[:])[::-1][:topN]
        top_attr_score = box_attr[top_attr]
        top_attrs[i] = top_attr
        top_attrs_scores[i] = top_attr_score

    # No need to add 1. just ATTRIBUTE[attr_idx]
    # where ATTRIBUTE[0] = __no_attribute__
    top_attrs = np.where(top_attrs_scores < attr_thresh, 0, top_attrs)
    top_attrs_scores = np.where(top_attrs == 0, 0.0, top_attrs_scores)

    return top_attrs, top_attrs_scores


def get_cls_boxes(st2_boxes, keep_box_indices, keep_box_cls_indices):
    boxes = st2_boxes[keep_box_indices].reshape(-1, 1601, 4)
    # shape [K, 4]
    final_boxes = np.zeros((boxes.shape[0], 4))

    for i in range(len(keep_box_cls_indices)):
        final_boxes[i] = boxes[i, keep_box_cls_indices[i]]

    return final_boxes


def get_cls_indices(st2_scores, keep_box_indices):
    # No need to add 1. just CLASSES[attr_idx] where CLASSES[0] = __background__
    # To get the class: CLASSES[cls_index]
    return np.argmax(st2_scores[keep_box_indices][:, 1:], axis=1) + 1


def load_img_paths(dir_path):
    img_paths = glob(os.path.join(dir_path, "*.jpg"))
    return img_paths


def image_id_from_path(image_path):
    """Given a path to an image, return its id.
    Parameters
    ----------
    image_path : str
        Path to image, e.g.: coco_train2014/COCO_train2014/000000123456.jpg
    Returns
    -------
    int
        Corresponding image id (123456)
    """

    return int(image_path.split("/")[-1][-16:-4])


"""
How to convert back to attr and cls to words, please 
see each functions and demo.ipynb.
"""


class Dataset:

    def __init__(self, num_boxes, num_images, split, topNattr):
        self.file_name = args.out_path

        self.save_h5 = h5py.File(self.file_name, 'a')
        self.save_h5.attrs['split'] = split

        self.datasets = {}
        min_boxes, max_boxes = num_boxes
        for field in ['image_id', 'image_h', 'image_w', 'num_boxes']:
            self.datasets[field] = self.save_h5.create_dataset(field, (num_images,), dtype='int')

        self.datasets['cls_indices'] = self.save_h5.create_dataset('cls_indices', (num_images, max_boxes), dtype='int')
        self.datasets['top_attrs'] = self.save_h5.create_dataset('top_attrs', (num_images, max_boxes, topNattr),
                                                                 dtype='int')
        self.datasets['top_attrs_scores'] = self.save_h5.create_dataset('top_attrs_scores',
                                                                        (num_images, max_boxes, topNattr),
                                                                        dtype='float32')
        self.datasets['boxes'] = self.save_h5.create_dataset('boxes', (num_images, max_boxes, 4), dtype='float32')
        self.datasets['features'] = self.save_h5.create_dataset('features', (num_images, max_boxes, 2048),
                                                                dtype='float32')
        self.datasets['st1_boxes'] = self.save_h5.create_dataset('st1_boxes', (num_images, max_boxes, 4),
                                                                 dtype='float32')

        # for f in FIELDNAMES:
        #     print(f, self.datasets[f].shape)

        self.cur_idx = 0

    def update(self, out):
        for key in out:
            if isinstance(out[key], np.ndarray):
                pass
                # print("key", key, out[key].shape)
            self.datasets[key][self.cur_idx] = out[key]
        self.cur_idx += 1
        self.save_h5.attrs['cur_idx'] = self.cur_idx

    def close(self):
        self.save_h5.close()


def pad(x, max_boxes):
    """
    :param x: [K, N]
    :param max_boxes: K needs to pad up to max_boxes
    :return:  [max_boxes, N]
    """
    num_boxes = x.shape[0]
    if num_boxes == max_boxes:
        return x
    if len(x.shape) == 1:
        return np.pad(x, (0, max_boxes - num_boxes), 'constant', constant_values=0)
    else:
        return np.pad(x, ((0, max_boxes - num_boxes), (0, 0)), 'constant', constant_values=0)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate bbox output from a Fast R-CNN network')
    parser.add_argument('--split', default="demo")
    parser.add_argument('--topNattr', default=20, type=int)
    parser.add_argument('--data_path', default="data/demo")
    parser.add_argument('--num_images', default=30, type=int)
    parser.add_argument('--out_path', default=None)
    parser.add_argument('--prototxt', default='models/vg/ResNet-101/faster_rcnn_end2end_final/test.prototxt')
    parser.add_argument('--weights')
    args = parser.parse_args()

    # Load model
    net = caffe.Net(args.prototxt, caffe.TEST, weights=args.weights)

    img_paths = load_img_paths(args.data_path)
    print("number of images", len(img_paths))
    datasets = [Dataset(num_boxes=num_boxes, num_images=args.num_images, split=args.split, topNattr=args.topNattr) for
                num_boxes in NUM_BOXES]

    with tqdm(total=len(img_paths)) as pbar:
        for img_path in img_paths:
            caffe.set_mode_gpu()
            caffe.set_device(0)
            preds = extract_im(net, img_path)

            for i, (keep_box_indices, keep_box_cls_indices) in enumerate(preds["keep_ind"]):
                # shape [K, topN]
                # shape [K, topN]
                top_attrs, top_attrs_scores = get_topN_attrs(preds["st2_attr_scores"],
                                                             keep_box_indices,
                                                             topN=20, attr_thresh=attr_thresh)
                # shape [K, 4]
                boxes = get_cls_boxes(preds["st2_boxes"],
                                      keep_box_indices=keep_box_indices,
                                      keep_box_cls_indices=keep_box_cls_indices)

                # shape [K, 2048]
                features = preds["pool5"][keep_box_indices]

                # shape [K, 4]
                st1_boxes = preds["st1_boxes"][keep_box_indices]

                # shape [K, ]
                cls_indices = get_cls_indices(preds['st2_scores'], keep_box_indices)

                min_boxes, max_boxes = NUM_BOXES[i]
                num_boxes = len(keep_box_indices)
                out = {
                    "image_id": preds["image_id"],
                    "image_h": preds["image_h"],
                    "image_w": preds["image_w"],
                    "num_boxes": num_boxes,
                    "cls_indices": pad(cls_indices, max_boxes),
                    "top_attrs": pad(top_attrs, max_boxes),
                    "top_attrs_scores": pad(top_attrs_scores, max_boxes),
                    "boxes": pad(boxes, max_boxes),
                    "features": pad(features, max_boxes),
                    "st1_boxes": pad(st1_boxes, max_boxes),
                }
                datasets[i].update(out)

            pbar.update(1)

        for i in range(len(datasets)):
            datasets[i].close()
