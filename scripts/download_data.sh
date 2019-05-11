#!/usr/bin/env bash


function download_file (){
    echo $1
    if [ -f $1 ]; then
        echo "The file '$1' exists. No need to download :3"
    else
        echo "The file '$FILE' in not found. Start downloading"
        wget -O $1 $2
    fi
}


FILE[1]='/content/datasets/features_faster_rcnn_x101_train.h5'
FILE[2]='/content/datasets/features_faster_rcnn_x101_val.h5'
FILE[3]='/content/datasets/features_faster_rcnn_x101_test.h5'
FILE[4]='/content/datasets/visdial_1.0_word_counts_train.json'
FILE[5]='/content/datasets/visdial_1.0_train.zip'
FILE[6]='/content/datasets/visdial_1.0_val.zip'
FILE[7]='/content/datasets/visdial_1.0_val_dense_annotations.json'

LINK[1]='https://s3.amazonaws.com/visual-dialog/data/v1.0/2019/features_faster_rcnn_x101_train.h5'
LINK[2]='https://s3.amazonaws.com/visual-dialog/data/v1.0/2019/features_faster_rcnn_x101_val.h5'
LINK[3]='https://s3.amazonaws.com/visual-dialog/data/v1.0/2019/features_faster_rcnn_x101_test.h5'
LINK[4]='https://s3.amazonaws.com/visual-dialog/data/v1.0/2019/visdial_1.0_word_counts_train.json'
LINK[5]='https://www.dropbox.com/s/ix8keeudqrd8hn8/visdial_1.0_train.zip?dl=0'
LINK[6]='https://www.dropbox.com/s/ibs3a0zhw74zisc/visdial_1.0_val.zip?dl=0'
LINK[7]='https://www.dropbox.com/s/3knyk09ko4xekmc/visdial_1.0_val_dense_annotations.json\?dl=0'

# Download
for i in {1..20}
do
    download_file ${FILE[$i]} ${LINK[$i]}
done

# Unzip
for i in {5..6}
do
    if [ -f ${FILE[i]} ]; then
        unzip ${FILE[i]}
        rm ${FILE[$i]}
    fi
done
