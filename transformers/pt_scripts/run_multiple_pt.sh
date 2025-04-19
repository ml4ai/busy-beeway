#!/bin/bash

filename=$1

counter=0
seed=$5
while read -r line; do
  XLA_PYTHON_CLIENT_MEM_FRACTION=0.50 CUDA_VISIBLE_DEVICES=$2 python run_pt_training.py "${3}/${line}_pref.hdf5" -o "${4}/${line}" -n 200 -b 256 -d 256 -w 0 -l 0.0 0.0001 0.0 -s $((counter + seed))
  counter=$((counter + 1))
done < $filename