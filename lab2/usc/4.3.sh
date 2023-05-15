#!/bin/bash
source ./path.sh

for dir in train test dev; do
    steps/make_mfcc.sh  --mfcc-config conf/mfcc.conf --cmd  "run.pl" data/$dir exp/make_mfcc/$dir mfcc_${dir}
    steps/compute_cmvn_stats.sh data/$dir exp/make_mfcc/$dir mfcc
done