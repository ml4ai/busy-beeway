#!/bin/bash

filename=$1

while read -r line; do
  CUDA_VISIBLE_DEVICES=$2 python combine_state_data.py "${1}" -e "${line}" -d "${3}" -s "${3}/bbway1_no_${line}.hdf5"
done < $filename