#!/usr/bin/env bash

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python train_matching.py --mode=train --encoder=bert --save_path=/data/mingyang/hisum --gpus=0,1,2,3,4,5,6,7 --batch_size=16
