#!/bin/bash
source ./path.sh

# make directories for mfcc statistics
mkdir mfcc_logs

for dir in train test dev; do
    steps/make_mfcc.sh data/$dir mfcc_logs/$dir mfcc_${dir}
    steps/compute_cmvn_stats.sh data/$dir mfcc_logs/$dir mfcc
done

rm -R mfcc_logs


# question 3

# frames per sentence
feat-to-len scp:data/train/feats.scp ark,t:data/train/feats.lengths
head -5 data/train/feats.lengths

# characteristics dimension 
feat-to-dim ark:mfcc_train/raw_mfcc_train.1.ark -