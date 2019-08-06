"""Prefer Python 2.7 to run these lines of code"""
import csv
import sys
import h5py
import base64
import argparse
import numpy as np
from tqdm import tqdm

csv.field_size_limit(sys.maxsize)

def get_num_images(path):
	with h5py.File(path, 'r') as f:
		return len(list(f['image_id']))

parser = argparse.ArgumentParser()
parser.add_argument("--in_tsv_file")
parser.add_argument("--out_h5_file")
parser.add_argument("--old_h5_file")
parser.add_argument("--split")
parser.add_argument("--max_boxes", default=36, type=int)
parser.add_argument("--feat_dims", default=2048, type=int)
args = parser.parse_args()

num_images = get_num_images(args.old_h5_file)
print("Total images", num_images)

# create an output HDF to save extracted features
FIELDNAMES = ['image_id', 'image_w', 'image_h', 'num_boxes', 'boxes', 'features']
SIZES = [(num_images,)] * 4 + [(num_images, args.max_boxes, 4)] + [(num_images, args.max_boxes, args.feat_dims)]
DTYPES = ['int'] * 4 + ['float32'] * 2

save_h5 = h5py.File(args.out_h5_file, "w")
datasets = {}

for field, size, dtype in zip(FIELDNAMES, SIZES, DTYPES):
	datasets[field] = save_h5.create_dataset(field, size, dtype=dtype)
save_h5.attrs["split"] = args.split

with tqdm(total=num_images) as pbar:
	with open(args.in_tsv_file, "r+b") as f:
		reader = csv.DictReader(f, delimiter='\t', fieldnames=FIELDNAMES)

		for i, item in enumerate(reader):

			for field in ['image_id', 'image_w', 'image_h', 'num_boxes']:
				datasets[field][i] = int(item[field])

			for field in ['boxes', 'features']:
				num_boxes = int(item['num_boxes'])
				item[field] = np.frombuffer(base64.decodestring(item[field]),
				                            dtype=np.float32).reshape((num_boxes, -1))
				datasets[field][i] = item[field]
			pbar.update(1)

save_h5.close()
