#!/bin/bash

# Bash script to run training on each of the 4 datasets: Cora, CiteSeer, PubMed, PPI

datasets=("Cora" "CiteSeer" "PubMed" "PPI")

for dataset in "${datasets[@]}"; do
    echo "Starting training on $dataset"
    python main.py --dataset $dataset
    echo "Finished training on $dataset"
done
