#!/bin/bash

# Define the list of input and output file pairs
lexicon_file=("./lexicon.txt")
output_file=("./data/local/dict/nonsilence_phones.txt")

non_silent_phonemes=()
# Loop through each line in the input file
while read -r line; do

    # take only phoneme sequence
    sentence=${line#* }

    # Split phoneme sequence into phonemes
    words=($sentence)
    
    for word in "${words[@]}"; do
        #if word not in list, append
        if [[ ! " ${non_silent_phonemes[@]} " =~ " ${word} " ]]; then
            non_silent_phonemes+=("${word} ")
            #echo ${word}
        fi
    done

done < "$lexicon_file"

# Write sorted phonemes to output file
echo -n "${non_silent_phonemes[@]}" | tr ' ' '\n' | sort -u | sed '/^\s*$/d' >> "${output_file}"