#!/bin/bash

filename=$1

while read -r line; do
  XLA_PYTHON_CLIENT_PREALLOCATE=false CUDA_VISIBLE_DEVICES=$2 python run_pt_training.py "${3}/${line}.hdf5" "${3}/bbway1_no_${line}.hdf5" -o "${4}/${line}" -n 200 -b 256 -d 256 -w 20 -l 0.0 0.0001 0.0
done < $filename