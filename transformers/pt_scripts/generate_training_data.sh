#!/bin/bash

filename=$1

counter=0
seed=$5
while read -r line; do
  CUDA_VISIBLE_DEVICES=$2 python create_pref_data.py "${1}" "${line}" -d "${3}" -s "${4}/${line}_pref.hdf5" -r $((counter + seed))
  counter=$((counter + 1))
done < $filename