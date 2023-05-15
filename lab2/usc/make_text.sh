#!/bin/bash

# Define the list of input and output file pairs
input_files=("./data/train/uttids" "./data/test/uttids" "./data/dev/uttids")
output_files=("./data/train/text" "./data/test/text" "./data/dev/text")

transcriptions_file=("./transcriptions.txt")

# Loop through each input/output file pair
for i in "${!input_files[@]}"; do

  # Loop through each line in the input file
  while read line; do

    #split to get utterance_id
    IFS='' read -r -a id <<< "$line"

    #split to take utterance number
    IFS='_' read -r -a strings <<< "$line"

    # Combine line number with "utterance_id"
    utterance_id="${id[0]}"

    search_string="${strings[1]}"

    # Search for lines where the first string is the search_string
    result=$(grep "^${search_string}" "${transcriptions_file}")

    # If a matching line is found, extract the rest of the string
    if [ -n "${result}" ]; then
        text=$(echo "${result}" | cut -d$'\t' -f2-)
    else
        echo "No matching line found"
    fi

    # Write the new line to the output file
    echo "$utterance_id $text" >> "${output_files[$i]}"

  done < "${input_files[$i]}"

done