#!/bin/bash

# command line arg
FST=${1}

# score variables
hits=0 #number of accurate predicts
N=0 # number of tests

# Read the first 20 lines of input
input=$(head -n 20 data/spell_test.txt)

# Loop through each line in the input
while read -r line; do
    words=$(echo "$line" | cut -d " " -f2-)
    first=$(echo "$line" | cut -d " " -f1 | cut -d ':' -f1)
    printf $first":"
    # Loop through each word in the input
    for word in $words; do
        printf " "
        ((N++))
        # Run the .sh script corresponding to the current word
        sh scripts/predict.sh $FST $word
        predicted=$(sh scripts/predict.sh "$FST" "$word")
        if [[ "$predicted" = "$first" ]]; then
            ((hits++))
        fi
    done
    printf "\n"
done <<< "$input"

accuracy=$(bc <<< "scale=2; $hits / $N")
printf "Accuracy: %s \n" "$accuracy"
