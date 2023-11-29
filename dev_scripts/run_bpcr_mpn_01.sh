#!/bin/bash

export CUDA_VISIBLE_DEVICES=$1

#scenes=("chair" "drums" "ficus" "hotdog" "lego" "materials" "mic" "ship")
scenes=("chair" "drums")

for scene in "${scenes[@]}"; do
    echo "------Training: device $1, scene $scene-------"
    #python run_rasterize.py --config=configs/bpcr_mpn_crop800_dim8/${scene}_pointnerf.txt
    python main.py --config=configs/bpcr_mpn_tiny_crop800_dim8/${scene}_pointnerf.txt
done
