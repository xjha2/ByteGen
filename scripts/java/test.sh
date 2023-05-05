#!/usr/bin/env bash

function make_dir () {
    if [[ ! -d "$1" ]]; then
        mkdir $1
    fi
}

SRC_DIR=../..
DATA_DIR=${SRC_DIR}/data/10w
MODEL_DIR=${SRC_DIR}/tmp

make_dir $MODEL_DIR

DATASET=java
CODE_EXTENSION=original_subtoken
JAVADOC_EXTENSION=original

function test () {

echo "============TESTING============"

RGPU=$1
MODEL_NAME=$2

PYTHONPATH=$SRC_DIR CUDA_VISIBLE_DEVICES=$RGPU python -W ignore ${SRC_DIR}/main/train.py \
--only_test True \
--data_workers 5 \
--dataset_name $DATASET \
--data_dir ${DATA_DIR}/ \
--model_dir $MODEL_DIR \
--model_name $MODEL_NAME \
--dev_src test/bcode_test.txt \
--dev_tgt test/tgt_test.txt \
--uncase True \
--max_src_len 384 \
--max_tgt_len 30 \
--max_examples -1 \
--test_batch_size 64

}

test $1 $2
