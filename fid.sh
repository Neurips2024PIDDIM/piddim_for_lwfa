#!/bin/bash

# Get the root folders from command line arguments
dir1=$1
dir2=$2

# Define the range of subdirectories
start=1
end=22

# Initialize the sum variable and the sum of squares variable
sum=0
sum_sq=0

# Initialize variables for max and min FID values and their corresponding subdirectories
max_fid=0
min_fid=1000000
max_subdir=""
min_subdir=""

# Initialize the count variable
count=0

# Loop through the subdirectories
for ((i=start; i<=end; i++))
do
    # Check if the subdirectory exists in both directories
    if [ -d "$dir1/$i" ] && [ -d "$dir2/$i" ]; then
        # Run the command for each subdirectory and capture the output
        output=$(python -m pytorch_fid --device cuda:1 "$dir1/$i" "$dir2/$i")
        
        # Extract the float number at the end after 'FID'
        fid=$(echo $output | awk '{print $NF}')
        
        # Add the number to the sum
        sum=$(echo "$sum + $fid" | bc -l)
        
        # Add the square of the number to the sum of squares
        sum_sq=$(echo "$sum_sq + ($fid * $fid)" | bc -l)
        
        # Check if this is a new maximum or minimum FID value and update variables accordingly
        if (( $(echo "$fid > $max_fid" | bc -l) )); then
            max_fid=$fid
            max_subdir="$i"
        fi
        
        if (( $(echo "$fid < $min_fid" | bc -l) )); then
            min_fid=$fid
            min_subdir="$i"
        fi
        
        # Increment the count variable
        count=$((count+1))
    fi
done

# Calculate the average
average=$(echo "$sum / $count" | bc -l)

# Calculate the variance
variance=$(echo "($sum_sq - ($sum * $sum) / $count) / ($count - 1)" | bc -l)

echo "Average FID: $average"
echo "Maximum FID: $max_fid (in subdirectory $max_subdir)"
echo "Minimum FID: $min_fid (in subdirectory $min_subdir)"
echo "Variance of FID: $variance"
