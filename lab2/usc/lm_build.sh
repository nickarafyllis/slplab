#!/bin/bash

source ./path.sh

# Define the list of input and output file pairs
input_files=("./data/local/dict/lm_train.txt" "./data/local/dict/lm_test.txt" "./data/local/dict/lm_dev.txt")
output_dict=("./data/local/lm_tmp/train" "./data/local/lm_tmp/test" "./data/local/lm_tmp/dev")

# Loop through each input
for i in "${!input_files[@]}"; do
    #unigrams
    build-lm.sh -i "${input_files[i]}" -n 1 -o "${output_dict[i]}_unigram.ilm.gz"
    #bigrams
    build-lm.sh -i "${input_files[i]}" -n 2 -o "${output_dict[i]}_bigram.ilm.gz"
done