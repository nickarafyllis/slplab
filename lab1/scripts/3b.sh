#!/bin/bash

# Read the frequent tokens and their frequencies from the file
frequent_tokens=$(cat vocab/words.vocab.txt)

# Initialize the index counter to 0
index=0

# add <epsilon> with index 0
echo -e "<eps>\t$index" > vocab/words.syms
index=$((index + 1))

# Loop over the tokens and assign an index to each one
while read -r token frequency; do
  echo -e "$token\t$index"
  index=$((index + 1))
done <<< "$frequent_tokens" >> vocab/words.syms
