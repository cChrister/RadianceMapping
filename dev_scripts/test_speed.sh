#!/bin/bash

export CUDA_VISIBLE_DEVICES=$1

python speeding.py --config=configs/debug/lego_pointnerf.txt
