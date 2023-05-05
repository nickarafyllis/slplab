#!/bin/bash

# Define the list of input and output file pairs
input_files=("./data/train/text" "./data/test/text" "./data/dev/text")
output_files=("./data/train/phonemes" "./data/test/phonemes" "./data/dev/phonemes")

lexicon_file=("./lexicon.txt")

# Loop through each input/output file pair
for i in "${!input_files[@]}"; do

    # Loop through each line in the input file
    while read -r line; do

        # Split line into utterance ID and sentence
        utterance_id=${line%% *}
        sentence=${line#* }

        # Make sentence uppercase
        sentence=$(echo "$sentence" | tr '[:lower:]' '[:upper:]')
        # Remove special characters except single quotes
        sentence=$(echo "$sentence" | tr -cd "[:alnum:][:space:]'\-'"| sed 's/-/ /g')
        # Split sentence into words
        words=($sentence)
        
        # Replace each word with its corresponding phonemes from the lexicon
        phonemes=()
        for word in "${words[@]}"; do
            #echo ${word}
            # Search for lines where the first string is the search_string
            result=$(grep "^${word}[[:space:]]" "${lexicon_file}")

            # If a matching line is found, extract the rest of the string
            if [ -n "${result}" ]; then
                phonemes+=$(echo "${result}" | cut -d$'\t' -f2-)" "
            else
                echo "No matching line found for word: " $word
            fi
        done

        # Write new sentence to output file
        echo "$utterance_id "sil" "${phonemes[@]}" "sil"" >> "${output_files[$i]}"

    done < "${input_files[$i]}"

done
