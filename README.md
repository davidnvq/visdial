Table of content
----------------------
  * [Setup and Environment](#setup-and-environment)
  * [Download Data](#download-data)
  * [Training](#training)
  * [Evaluation](#evaluation)
  * [Result of Checkpoints](#result-of-checkpoints)
  * [Acknowledgements](#acknowledgements)

This is the code implementation for the paper titled: "**Efficient Attention Mechanism for Visual Dialog that can Handle All the Interactions between Multiple Inputs**" (**Accepted to ECCV 2020**).
If you find this code useful or use our method as the baseline for comparison, please kindly cite the paper with the following bibtex or the plain citation:

```
@article{nguyen2019efficient,
  title={Efficient Attention Mechanism for Handling All the Interactions between Many Inputs with Application to Visual Dialog},
  author={Nguyen, Van-Quang and Suganuma, Masanori and Okatani, Takayuki},
  journal={arXiv preprint arXiv:1911.11390},
  year={2019}
  
or as simply plain as:

Nguyen, Van-Quang, Masanori Suganuma, and Takayuki Okatani. "Efficient Attention Mechanism for Handling All the Interactions between Many Inputs with Application to Visual Dialog." arXiv preprint arXiv:1911.11390 (2019).

```

Setup and Environment
----------------------
This code is implemented using the following environment configurations:

Component      	|  Details  	|
 ------- 	| ------ 	|
 Pytorch 	| version 1.2 	|
 Python 	| version 3.7 	|
 GPU    	|  Tesla V100-SXM2 (16GB) |
No. of GPUs 	| 4 		|
  CUDA 		| 10.0 		|
  GPU Driver 	| 410.104 	| 
  RAM  		| 376GB 	|
  CPU 		| Xeon(R) Gold 6148 CPU @ 2.40GHz|
  

To set up the environment, we recommend you to set up a virtual environment using Anaconda.

1. Install Anaconda or Miniconda distribution based on Python3+ from their [downloads' site](https://www.anaconda.com/distribution/).
2. Clone this repository and create an environment
3. Install all the dependencies

```sh
conda create -n visdial python=3.7

# activate the environment and install all dependencies
conda activate visdial

# Install the dependencies
export PROJ_ROOT='/path/to/visualdialog/'
cd $PROJ_ROOT/
pip install -r requirements.txt
```

Download Data
-------------

1. Download the following `json` files for VisDial v1.0 and put them in `$PRO_ROOT/dataset/annotations/`:
    * For training set [here](https://www.dropbox.com/s/ix8keeudqrd8hn8/visdial_1.0_train.zip?dl=0).
    * For validation set [here](https://www.dropbox.com/s/ibs3a0zhw74zisc/visdial_1.0_val.zip?dl=0) 
    as well as the [Dense answer annotations](https://www.dropbox.com/s/3knyk09ko4xekmc/visdial_1.0_val_dense_annotations.json?dl=0).
    * For test set [here](https://www.dropbox.com/s/o7mucbre2zm7i5n/visdial_1.0_test.zip?dl=0).
    
    
2. Download the following `json` files for VisDial v0.9 and also put them in `$PROJ_ROOT/dataset/annotations/`:
    * For training set [here](https://s3.amazonaws.com/visual-dialog/v0.9/visdial_0.9_train.zip).
    * For validation set [here](https://s3.amazonaws.com/visual-dialog/v0.9/visdial_0.9_val.zip).


2. Get the word counts for VisDial v1.0 train split [here](https://s3.amazonaws.com/visual-dialog/data/v1.0/2019/visdial_1.0_word_counts_train.json) and put it in `$PROJ_ROOT/dataset/annotations/`. They are used to build the vocabulary.

3. Get the image features. We use the extracted features for VisDial v1.0 images using a Faster-RCNN pre-trained on Visual Genome. 

    * First download images for training set of Visdial v1.0 from COCO train2014 and val2014, which are available [here](http://cocodataset.org/#download) and also download the images for validation and test sets of Visdial v1.0 from [here](https://www.dropbox.com/s/twmtutniktom7tu/VisualDialog_val2018.zip?dl=0) 
    and [here](https://www.dropbox.com/s/mwlrg31hx0430mt/VisualDialog_test2018.zip?dl=0).
    * Then follow the instruction [here](https://github.com/peteanderson80/bottom-up-attention) to extract the bottom-up-attention features for images based on the pretrained Faster-RCNN:
        * First, clone the code provided by the authors at https://github.com/peteanderson80/bottom-up-attention. 
        * Second, setup the environment as [here](https://github.com/peteanderson80/bottom-up-attention). 
        * Then, extract the features as mentioned in our paper. We provide our code for extraction; please copy the code `$PROJ_ROOT/others/generate_visdial.py` from our project to `bottom-up-attention/tools`.
        * Run the following command to extract:
        ```sh
        # Estimate 10 hours
        # Extract the image features for the training split
        /usr/bin/python generate_visdial.py \
        --split "train" \
        --topNattr 20 \
        --num_images 123287 \
        --data_path '/path_to_the_image_dir/trainval2014' \
        --out_path '$PROJ_ROOT/datasets/bottom-up-attention/trainval_resnet101_faster_rcnn_genome_num_boxes_100.h5' \
        --prototxt 'models/vg/ResNet-101/faster_rcnn_end2end_final/test.prototxt' \
        --weights '/path_to_bottom_up_attention_checkpoints/bottom-up-attention/resnet101_faster_rcnn_final.caffemodel'
      
        # Estimate 35 minutes
        # Extract the image features for the validation split
        /usr/bin/python generate_visdial.py \
        --split "val" \
        --topNattr 20 \
        --num_images 2064 \
        --data_path '/path_to_the_image_dir/VisualDialog_val2018' \
        --out_path '/$PROJ_ROOT/datasets/bottom-up-attention/val2018_resnet101_faster_rcnn_genome_num_boxes_100.h5' \
        --prototxt 'models/vg/ResNet-101/faster_rcnn_end2end_final/test.prototxt' \
        --weights '/path_to_bottom_up_attention_checkpoints/bottom-up-attention/resnet101_faster_rcnn_final.caffemodel'     
        
        # Estimate 2 hours
        # Extract the image features for the test split
        /usr/bin/python generate_visdial.py \
        --split "test" \
        --topNattr 20 \
        --num_images 8000 \
        --data_path '/path_to_the_image_dir/VisualDialog_test2018' \
        --out_path '/$PROJ_ROOT/datasets/bottom-up-attention/test2018_resnet101_faster_rcnn_genome_num_boxes_100.h5' \
        --prototxt 'models/vg/ResNet-101/faster_rcnn_end2end_final/test.prototxt' \
        --weights '/path_to_bottom_up_attention_checkpoints/bottom-up-attention/resnet101_faster_rcnn_final.caffemodel'     
        ```
      * At the end, the directory `$PROJ_ROOT/datasets/bottom-up-attention/` should have the following files:
      ```sh
      ./trainval_resnet101_faster_rcnn_genome_100.h5
      ./val2018_resnet101_faster_rcnn_genome_100.h5
      ./test2018_resnet101_faster_rcnn_genome_100.h5
      ```
     * In the `$PROJ_ROOT/datasets/`, we also provide the available data that you need:
        ```
        $PROJ_ROOT/datasets/glove/embedding_Glove_840_300d.pkl
        $PROJ_ROOT/datasets/genome/1600-400-20/attributes_vocab.txt
        $PROJ_ROOT/datasets/genome/1600-400-20/objects_vocab.txt
        ```

Training
--------

Our code supports both generative and discriminative decoders (and both of them that we call `misc`).
We also provide the training script which supports Visdial v1.0 and Visdial v0.9.

**Note**: If the CUDA is out of memory, please consider to decrease the `batch_size`.

### Training on Visdial v1.0 
To reproduce our results on Visdial v1.0, please run the following command (the other hyperparameters will be considered as default as our paper's):

```sh
CUDA_VISIBLE_DEVICES=0,1,2,3 python train.py \
--config_name model_v10 \
--save_dir checkpoints \
--batch_size 8 \
--decoder_type misc \
--init_lr 0.001 \
--scheduler_type "LinearLR" \
--num_epochs 15 \
--num_samples 123287 \
--milestone_steps 3 5 7 9 11 13 \
--encoder_out 'img' 'ques' \
--dropout 0.1 \
--img_has_bboxes \
--ca_has_layer_norm \
--ca_num_attn_stacks 2 \
--ca_has_residual \
--ca_has_self_attns \
--txt_has_layer_norm \
--txt_has_decoder_layer_norm \
--txt_has_pos_embedding \
```
**Note 1**: The `batch_size` is set per each GPU. If you have 4 GPUs, the number of actual `batch_size` is 32 as ours.

**Note 2**: You can also train 
a discriminative model or a generative model by specifying `--decoder_type` as `disc` and `gen`, respectively.

### Training on Visdial v0.9
To reproduce our results on Visdial v0.9, please run the following command (the other hyperparameters will be considered as default as our paper's):

```sh
CUDA_VISIBLE_DEVICES=0,1,2,3 python train.py \
--config_name misc_v0.9 \
--save_dir checkpoints \
--v0.9 \
--batch_size 8 \
--decoder_type misc \
--init_lr 0.001 \
--scheduler_type "LinearLR" \
--num_epochs 5 \
--num_samples 123287 \
--milestone_steps 3 5 \
--encoder_out 'img' 'ques' \
--dropout 0.1 \
--img_has_bboxes \
--ca_has_layer_norm \
--ca_num_attn_stacks 2 \
--ca_has_residual \
--ca_has_self_attns \
--txt_has_layer_norm \
--txt_has_decoder_layer_norm \
--txt_has_pos_embedding \
--val_feat_img_path "datasets/bottom-up-attention/trainval_resnet101_faster_rcnn_genome_num_boxes_100.h5" \
--train_feat_img_path "datasets/bottom-up-attention/trainval_resnet101_faster_rcnn_genome_num_boxes_100.h5" \
--val_json_dialog_path "datasets/annotations/visdial_0.9_val.json" \
--train_json_dialog_path "datasets/annotations/visdial_0.9_val.json"
```
**Note 1**: You must turn the flag `--v0.9` on, then the corresponding `VisdialDataset` for Visdial v0.9 will be generated.

**Note 2**: `val_json_dialog_path` is the same as `train_feat_img_path` 
since the v0.9 validation split is part of `trainval` split in Visdial v1.0. It will not cause any confliction 
since the validation split v0.9 will be generated based on the `image_ids` we get from `val_json_dialog_path`.

As the original testbed, we also provide an `--overfit` flag, which can be useful for debugging.

### Saving model checkpoints
The checkpoint is saved at every epoch at the directory you specify with `--save_dir`. The default directory is `checkpoint/`.

### Logging

Tensorboard is used for logging training progress. Please go to `checkpoints/tensorboard` directory execute the following 
 ```shell script
 tensorboard --logdir ./ --port 8008
#  and open `localhost:8008` in the browser.
```

### Finetuning for Ensemble
Run the following command to perform fintuning:

```
python finetune.py \
--model_path path/to/checkpoint/model_v10.pth \
--save_path path/to/saved/checkpoint \
```
Evaluation
----------

The evaluation of a trained model checkpoint on the validation set can be done as follows:

```sh
python evaluate.py \
--model_path 'checkpoints/model_v10.pth' \
--split val \
--decoder_type disc \
--device 'cuda:0' \
--output_path 'checkpoints/val_v1_disc.json'
```
**Note 1**: You can evaluate on three kinds of decoders: `disc`, `gen`, and `misc`.

**Note 2**: The above script is also applicable for the `test` split by changing the value of `--split` to `test`. After that, 
please submit the `test_v1_disc.json` to the server for further evaluation.

**Note 3**: The above script is also applicable for the evaluation on Visdial v0.9.


This will generate an EvalAI submission file, and report metrics (Mean reciprocal rank, R@{1, 5, 10}, Mean rank), and Normalized Discounted Cumulative Gain (NDCG), introduced in the first Visual Dialog Challenge (in 2018).

Result of Checkpoints
----------------------------------
### The overall architecture

The get the summary of the overall architecture, run the following python code:

```python
import torch

model = torch.load('checkpoints/model_v10.pth')
print(model)
```

### The number of the stack of attention blocks
To compute the number of parameters in our proposed attention stacks, run the python code as follows:

```python
import torch
from visdial.utils import get_num_params

model = torch.load('checkpoints/model_v10.pth')
# The number of parameters per one stack
print(get_num_params(model.encoder.attn_encoder.cross_attn_encoder[0]))

# The number of parameters of the attention encoder
print(get_num_params(model.encoder.attn_encoder))
```


Performance on `v1.0 validation` split (trained on `v1.0` train + val):

  Model  |  R@1   |  R@5   |  R@10  | MeanR  |  MRR   |  NDCG  |
 ------- | ------ | ------ | ------ | ------ | ------ | ------ |
[model-v1.0] with outputs from disc | 0.4894 | 0.7865 | 0.8788 | 4.8589 | 0.6232| 0.6272 |
[model-v1.0] with outputs from gen  | 0.4044 | 0.6161 |	0.6971 | 14.9274| 0.5074| 0.6358 |
[model-v1.0] with outputs from the avg of two  |  0.4303|0.6663  | 0.7567 | 10.6030| 0.5436| 0.6575 |

Performance on `v1.0 test` split  (trained on `v1.0` train + val):

  Model  |  R@1   |  R@5   |  R@10  | MeanR  |  MRR   |  NDCG  |
 ------- | ------ | ------ | ------ | ------ | ------ | ------ |
[disc-model-v1.0] | 0.4700 | 0.7703 | 0.8775 | 4.90| 0.6065| 0.6092 |

Performance on `v0.9 validation` split (trained on `v0.9` train):

  Model  |  R@1   |  R@5   |  R@10  | MeanR  |  MRR   |
 ------- | ------ | ------ | ------ | ------ | ------ |
[disc-model-v0.9] | 55.05 | 0.83.98 | 91.58 | 3.69 | 67.94 |


Acknowledgements
----------------

* This code is built upon the fork of [visdial-challenge-starter-pytorch](https://github.com/batra-mlp-lab/visdial-challenge-starter-pytorch) 
developed by the team of researchers from Machine Learning and Perception Lab, Georgia Tech 
for [Visual Dialog Challenge 2019](https://visualdialog.org/challenge/2019).
We would like to thank them for providing this testbed. 


