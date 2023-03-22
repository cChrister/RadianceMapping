#!/bin/bash
#########################################################################
# File Name: run_training.sh
# Author: steve
# E-mail: yqykrhf@163.com
# Created Time: Tue 07 Mar 2023 11:12:53 PM CST
# Brief: 
#########################################################################
echo "------Training: $1, device $2-------"
export CUDA_VISIBLE_DEVICES=$2
python main.py --config=configs/$1.txt
