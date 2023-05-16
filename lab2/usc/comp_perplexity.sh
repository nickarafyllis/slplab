#!/bin/bash
source ./path.sh

echo "Perplexity of unigram model of validation set :"
compile-lm ./data/local/lm_tmp/dev_unigram.ilm.gz --eval=./data/local/dict/lm_dev.txt --dub=10000000

echo "Perplexity of bigram model of validation set :"
compile-lm ./data/local/lm_tmp/dev_bigram.ilm.gz --eval=./data/local/dict/lm_dev.txt --dub=10000000

echo "Perplexity of unigram model of test set :"
compile-lm ./data/local/lm_tmp/test_unigram.ilm.gz --eval=./data/local/dict/lm_test.txt --dub=10000000

echo "Perplexity of bigram model of test set :"
compile-lm ./data/local/lm_tmp/test_bigram.ilm.gz --eval=./data/local/dict/lm_test.txt --dub=10000000