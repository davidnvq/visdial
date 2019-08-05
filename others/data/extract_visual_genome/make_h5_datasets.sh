#!/usr/bin/env bash

/usr/bin/python make_h5_datasets.py \
--in_tsv_file /home/quanguet/datasets/visdial/bottom-up/val2018_resnet101_faster_rcnn_genome.tsv.0 \
--out_h5_file /home/quanguet/datasets/visdial/bottom-up/val2018_resnet101_faster_rcnn_genome_36.h5 \
--old_h5_file /home/quanguet/datasets/visdial/features_faster_rcnn_x101_val.h5 \
--split val \
--max_boxes 36 \
--feat_dims 2048