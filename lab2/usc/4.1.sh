#!/bin/bash

#4.1.1
cd ../ #go to egs dir
cp wsj/s5/path.sh usc
cp wsj/s5/cmd.sh usc

#4.1.2
cd usc
ln -s ../wsj/s5/steps ./steps
ln -s ../wsj/s5/utils ./utils

#4.1.3
#maybe do this in data dir
mkdir local
ln -s ../steps/score_kaldi.sh ./local/score_kaldi.sh

#4.1.4
mkdir conf
wget https://raw.githubusercontent.com/slp-ntua/slp-labs/master/lab2/mfcc.conf -P conf/

#4.1.5
mkdir data/lang
mkdir data/local/dict
mkdir data/local/lm_tmp
mkdir data/local/nist_lm