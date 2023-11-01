#!/bin/bash

echo "Run mpn after radiance mapping on nerf_synthetic"
export CUDA_VISIBLE_DEVICES=$1

scenes=("chair" "drums" "ficus" "hotdog" "lego" "materials" "mic" "ship")

for scene in "${scenes[@]}"; do
    echo "------Training: device $1, scene $scene-------"
    python run_rasterize.py --config=configs/mpn_rdmp/${scene}_pointnerf.txt
    python main.py --config=configs/mpn_rdmp/${scene}_pointnerf.txt
done
