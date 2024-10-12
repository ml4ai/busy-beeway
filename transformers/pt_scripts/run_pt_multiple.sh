#!/bin/bash

filename=$1

while read -r line; do
  CUDA_VISIBLE_DEVICES=$2 python run_pt_training.py "${3}/${line}.hdf5" -o "${4}/${line}" -n 10 -b 8 -d 256 -w 8 -l 0.0001 0.001 0.0001
done < $filename
