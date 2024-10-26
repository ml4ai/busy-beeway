#!/bin/bash

filename=$1

while read -r line; do
  CUDA_VISIBLE_DEVICES=$2 python generate_return_samples.py "${3}/${line}/best_model.pkl" "${4}" -t "${line}"
done < $filename
