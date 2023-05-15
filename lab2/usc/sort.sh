#!/bin/bash

# sort text wav.scp and utt2spk in all 3 folders
sort -o ./data/train/text ./data/train/text 
sort -o ./data/train/wav.scp ./data/train/wav.scp 
sort -o ./data/train/utt2spk ./data/train/utt2spk

sort -o ./data/test/text ./data/test/text 
sort -o ./data/test/wav.scp ./data/test/wav.scp 
sort -o ./data/test/utt2spk ./data/test/utt2spk

sort -o ./data/dev/text ./data/dev/text 
sort -o ./data/dev/wav.scp ./data/dev/wav.scp 
sort -o ./data/dev/utt2spk ./data/dev/utt2spk