#!/bin/bash

# Read the first 20 lines of input
input=$(head -n 20 data/spell_test.txt)

# Loop through each line in the input
while read -r line; do
    words=$(echo "$line" | cut -d " " -f2-)
    first=$(echo "$line" | cut -d " " -f1)
    printf $first 
    # Loop through each word in the input
    for word in $words; do
        printf " "
        # Run the .sh script corresponding to the current word
        sh scripts/predict.sh fsts/S.binfst $word
    done
    printf "\n"
done <<< "$input"
