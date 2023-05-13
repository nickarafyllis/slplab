#!/bin/bash

# Define the list of input and output file pairs
input_files=("./data/train/text" "./data/test/text" "./data/dev/text")
output_files=("./data/local/dict/lm_train.txt" "./data/local/dict/lm_test.txt" "./data/local/dict/lm_dev.txt")

lexicon_file=("./lexicon.txt")

# Loop through each input/output file pair
for i in "${!input_files[@]}"; do

    # Loop through each line in the input file
    while read -r line; do

        # Split line into utterance ID and sentence
        #utterance_id=${line%% *}
        sentence=${line#* }

        # Write new sentence to output file
        echo " <s> "${sentence}" </s>" >> "${output_files[$i]}"

    done < "${input_files[$i]}"

done