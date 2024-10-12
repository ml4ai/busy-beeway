#!/bin/bash

filename=$1

while read -r line; do
  CUDA_VISIBLE_DEVICES=$2 python run_pt_training.py "${3}/${line}.hdf5" -o "${4}/${line}" -n 30 -b 4 -d 256 -w 4 -l 0.00001 0.0001 0.00001
done < $filename
