#!/bin/bash

# Concatenate the two input files
sed '1s/^/\n/' "$2" > temp_file

cat "$1" temp_file > combined.txt

# Loop over each line in the combined file
while read phonem; do
  # Write the phonem followed by a space and then the same phonem
  echo "$phonem $phonem" >> ./data/local/dict/lexicon.txt
done < combined.txt

# Remove the temp files
rm combined.txt
rm temp_file