#!/bin/bash
#########################################################################
# File Name: run_training.sh
# Author: steve
# E-mail: yqykrhf@163.com
# Created Time: Tue 07 Mar 2023 11:12:53 PM CST
# Brief: 
#########################################################################
echo "------Training: debug/lego_pointnerf, device $1-------"
export CUDA_VISIBLE_DEVICES=$1
#python run_rasterize.py --config=configs/debug/lego_pointnerf.txt
python main.py --config=configs/debug/lego_pointnerf.txt
