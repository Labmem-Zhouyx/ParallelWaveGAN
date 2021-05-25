#!/bin/bash

config_pat=./egs/bznsyp/voc1/bznsyp_params.yaml
checkpoint=./egs/bznsyp/voc1/checkpoints/checkpoint-1000000steps.pkl
dumpdir=path/inference_mels/
outdir=path/inference_wavs/
if [ ! -d ${outdir}  ];then
  mkdir ${outdir}
fi

CUDA_VISIBLE_DEVICES=0 python ./parallel_wavegan/bin/decode.py --config ${config_path} --dumpdir ${dumpdir} --outdir ${outdir} --checkpoint ${checkpoint}
