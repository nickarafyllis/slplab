#!/bin/bash

# Define the list of input and output file pairs
input_files=("./data/train/uttids" "./data/test/uttids" "./data/dev/uttids")
output_files=("./data/train/utt2spk" "./data/test/utt2spk" "./data/dev/utt2spk")

# Loop through each input/output file pair
for i in "${!input_files[@]}"; do

  # Loop through each line in the input file
  while read line; do

    # Take the first two characters of the line
    speaker_id="${line:0:2}"

    # Combine line number with "utterance_id"
    utterance_id="${line:0:6}"

    # Write the new line to the output file
    echo "$utterance_id $speaker_id" >> "${output_files[$i]}"

  done < "${input_files[$i]}"

done