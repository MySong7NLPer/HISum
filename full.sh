#!/usr/bin/env bash

CUDA_VISIBLE_DEVICES=4,5,6,7 python train_matching.py --mode=train --encoder=bert --save_path=/data/mingyang/www/hyperbolic2 --gpus=0,1,2,3 --batch_size=8
#CUDA_VISIBLE_DEVICES=0,1,2,3 python train_matching.py --mode=train --encoder=bert --save_path=/data/mingyang/www/hyperbolic --gpus=0,1,2,3 --batch_size=8
