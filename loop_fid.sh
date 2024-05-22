#!/bin/bash

# Get the path to the results directory
results_dir=$1

# Loop over every folder in the results directory
for folder in "$results_dir"/*; do
    # Check if it's a directory
    if [ -d "$folder" ]; then
        # Run the fid.sh script and capture the output
        output=$(./fid.sh "data/with_gain" "$folder")
        
        # Extract the average FID value
        average_fid=$(echo "$output" | grep "Average FID" | awk '{print $NF}')
        
        # Append the folder name and average FID value to metrics.csv
        echo "$(basename $folder),$average_fid" >> metrics.csv
    fi
done
