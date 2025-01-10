#!/bin/bash

filename=$1

count=0
while read -r line; do
  #CUDA_VISIBLE_DEVICES=$2 python generate_return_samples.py "${3}/${line}/best_model.pkl" "${4}" -o "${5}" -t "${line}" &
  (( count ++ )) 
  if (( count = 10 )); then
      #wait
      count=0
  fi
done < $filename
