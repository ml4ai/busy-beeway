#!/bin/bash

filename=$1

while read -r line; do
  CUDA_VISIBLE_DEVICES=$2 python run_vt_training.py "${3}/${line}.hdf5" -o "${4}/${line}" -n 1000 -b 4 -d 8 -w 8 -l 0.00001 0.001 0.00001
done < $filename
