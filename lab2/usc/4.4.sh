#!/bin/bash
source ./path.sh

# 4.4.1
# train monophone GMM-HMM model
steps/train_mono.sh data/train data/lang exp/mono

# 4.4.2
# create HCLG graph for unigram and bigram
utils/mkgraph.sh data/lang_phones_ug exp/mono exp/mono_graph_ug
utils/mkgraph.sh data/lang_phones_bg exp/mono exp/mono_graph_bg

# 4.4.3
# decode validation and test data with Viterbi
for dir in test dev; do
    steps/decode.sh exp/mono_graph_ug data/$dir exp/mono/decode_${dir}_ug
    steps/decode.sh exp/mono_graph_bg data/$dir exp/mono/decode_${dir}_bg
done

# 4.4.5
# align phones using monophone model
steps/align_si.sh data/train data/lang exp/mono exp/mono_ali

# train triphone model
steps/train_deltas.sh 2000 10000 data/train data/lang exp/mono_ali exp/tri1

# create HCLG graph for unigram and bigram
utils/mkgraph.sh data/lang_phones_ug exp/tri1 exp/tri1_graph_ug
utils/mkgraph.sh data/lang_phones_bg exp/tri1 exp/tri1_graph_bg

# decode validation and test data with Viterbi
for dir in test dev; do
    steps/decode.sh exp/tri1_graph_ug data/$dir exp/tri1/decode_${dir}_ug
    steps/decode.sh exp/tri1_graph_bg data/$dir exp/tri1/decode_${dir}_bg
done
