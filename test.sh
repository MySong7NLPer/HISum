#!/usr/bin/env bash



CUDA_VISIBLE_DEVICES=7 python train_matching.py --mode=test --encoder=bert --save_path=/data/mingyang/www/hyperbolic2/2023-02-09-22-52-49-518193 --gpus=0
#CUDA_VISIBLE_DEVICES=1 python train_matching.py --mode=test --encoder=roberta --save_path=/data/mingyang/matchsum_new/cls_match_adaptive/2021-11-27-17-39-59-085717 --gpus=0






