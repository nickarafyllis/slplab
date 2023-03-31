#!/bin/bash

# Read the frequent tokens and their frequencies from the file
frequent_tokens=$(cat vocab/words.vocab.txt)

# Initialize the index counter to 0
index=0

# Loop over the tokens and assign an index to each one
while read -r token frequency; do
  echo -e "$token\t$index"
  index=$((index + 1))
done <<< "$frequent_tokens" > vocab/words.syms
