#!/bin/bash

for folder in run_*; do
    cp "$folder/hierarchical_model_corner.png" "corner-plot/hierarchical_model_corner_$folder.png"
done
