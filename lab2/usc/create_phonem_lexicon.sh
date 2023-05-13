#!/bin/bash

# Concatenate the two input files
cat "$1" "$2" > combined.txt

# Loop over each line in the combined file
while read phonem; do
  # Write the phonem followed by a space and then the same phonem
  echo "$phonem $phonem" >> ./data/local/dict/lexicon.txt
done < combined.txt

# Remove the temp files
rm combined.txt