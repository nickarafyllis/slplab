#!/bin/bash

# Read the text file into a variable
text=$(cat corpus.txt)

# Tokenize the text using grep 
tokens=$(echo "$text" | grep -oE '\w+')

# Count the frequency of each token using sort and uniq
counts=$(echo "$tokens" | sort | uniq -c)

# Filter the tokens that occur more than 5 times
frequent_tokens=$(echo "$counts" | awk '$1 > 5 {print $2 "\t" $1}')

# Create the output directory if it doesn't exist
mkdir -p vocab

# Write the frequent tokens to a file
echo "$frequent_tokens" > vocab/words.vocab.txt
