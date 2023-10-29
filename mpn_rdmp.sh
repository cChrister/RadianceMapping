#!/bin/bash
#########################################################################
# File Name: run_training.sh
# Author: steve
# E-mail: yqykrhf@163.com
# Created Time: Tue 07 Mar 2023 11:12:53 PM CST
# Brief: 
#########################################################################

export CUDA_VISIBLE_DEVICES=$1

scenes=("chair" "drums" "ficus" "hotdog" "lego" "materials" "mic" "ship")

for scene in "${scenes[@]}"; do
    echo "------Training: device $1, scene $scene-------"
    # python run_rasterize.py --config=configs/mpn_rdmp/${scene}_pointnerf.txt
    python main.py --config=configs/mpn_rdmp/${scene}_pointnerf.txt
done
