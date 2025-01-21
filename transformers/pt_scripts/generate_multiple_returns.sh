#!/bin/bash

filename=$1

while read -r line; do
  nice python generate_return_samples.py "${2}/${line}/best_model.pkl" "${3}" -p -f 50 -j -d "${4}" -e "${5}" -o "${6}" 
done < $filename