#!/bin/bash

# Command line args
DICTIONARY=${1}
WORDS=${2}

# Read the frequent tokens and their frequencies from the file
frequent_tokens=$(cat "$DICTIONARY")

# Initialize the index counter to 0
index=0

# add <epsilon> with index 0
echo -e "<eps>\t$index" > $WORDS
index=$((index + 1))

# Loop over the tokens and assign an index to each one
while read -r token frequency; do
  echo -e "$token\t$index"
  index=$((index + 1))
done <<< "$frequent_tokens" >> $WORDS
