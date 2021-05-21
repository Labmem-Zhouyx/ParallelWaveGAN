#!/bin/bash

config_path=./egs/bznsyp/voc1/bznsyp_params.yaml
train_dump_dir=/data/datasets/vocoder_BZNSYP/train_data/
dev_dump_dir=/data/datasets/vocoder_BZNSYP/val_data/
out_dir=./egs/bznsyp/voc1/checkpoints/

CUDA_VISIBLE_DEVICES=0 python ./parallel_wavegan/bin/train.py --config ${config_path} --train-dumpdir ${train_dump_dir} --outdir ${out_dir} --dev-dumpdir ${dev_dump_dir}
