# Validation
./tools/generate_rpn_tsv.py --gpu 0 \
--cfg experiments/cfgs/faster_rcnn_end2end_resnet.yml \
--def models/vg/ResNet-101/faster_rcnn_end2end_final/test_rpn.prototxt \
--out /home/quanguet/datasets/bottom-up/val2018_resnet101_faster_rcnn_genome.tsv \
--net data/resnet101_faster_rcnn_final_iter_320000.caffemodel \
--split /home/quanguet/datasets/visdial/VisualDialog_val2018

# Test
./tools/generate_rpn_tsv.py --gpu 0 \
--cfg experiments/cfgs/faster_rcnn_end2end_resnet.yml \
--def models/vg/ResNet-101/faster_rcnn_end2end_final/test_rpn.prototxt \
--out /home/quanguet/datasets/bottom-up/test2018_resnet101_faster_rcnn_genome.tsv \
--net data/resnet101_faster_rcnn_final_iter_320000.caffemodel \
--split /home/quanguet/datasets/visdial/VisualDialog_test2018

## # Train
#./tools/generate_tsv.py --gpu 0 \
#--cfg experiments/cfgs/faster_rcnn_end2end_resnet.yml \
#--def models/vg/ResNet-101/faster_rcnn_end2end_final/test.prototxt \
#--out /home/quanguet/datasets/bottom-up/trainval2014_resnet101_faster_rcnn_genome.tsv \
#--net data/resnet101_faster_rcnn_final_iter_320000.caffemodel \
#--split /home/quanguet/datasets/visdial/trainval2014