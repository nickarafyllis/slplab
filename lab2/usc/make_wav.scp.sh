#!/bin/bash

# Define the list of input and output file pairs
input_files=("./data/train/uttids" "./data/test/uttids" "./data/dev/uttids")
output_files=("./data/train/wav.scp" "./data/test/wav.scp" "./data/dev/wav.scp")

# Loop through each input/output file pair
for i in "${!input_files[@]}"; do

  # Initialize line counter
  line_num=1

  # Loop through each line in the input file
  while read line; do

    # Split the line into an array of strings
    IFS=' ' read -r -a strings <<< "$line"

    # Combine line number with "utterance_id"
    utterance_id="utterance_id_${line_num}"
    path="./wav/${strings[0]}.wav"

    # Write the new line to the output file
    echo "$utterance_id $path" >> "${output_files[$i]}"

    # Increment the line number counter
    line_num=$((line_num+1))

  done < "${input_files[$i]}"

done