#!/bin/bash

# Command line args
DICTIONARY=${1}
WORDS=${2}

# 9.b
# make acceptor and save it as W.fst
python3 scripts/make_acceptor_word_freq.py $DICTIONARY >./fsts/W.fst

# compile model to bin and save it as W.binfst
fstcompile --isymbols=./$WORDS --osymbols=./$WORDS ./fsts/W.fst ./fsts/W.binfst

# optimize model and save it as W_opt.binfst
fstrmepsilon fsts/W.binfst | fstdeterminize | fstminimize >fsts/W_opt.binfst

#9.c
# LV already exists, it is S
# sort outputs of transducer
fstarcsort --sort_type=olabel fsts/S.binfst fsts/LV_sorted.binfst

# sort inputs of acceptor
fstarcsort --sort_type=ilabel fsts/W_opt.binfst fsts/W_sorted.binfst

# compose transducer and acceptor to create min edit distance spell checker with word frequencies
fstcompose fsts/LV_sorted.binfst fsts/W_sorted.binfst fsts/LVW.binfst

#9.d
# EV already exists
# sort outputs of transducer
fstarcsort --sort_type=olabel fsts/EV.binfst fsts/EV_sorted.binfst

# compose transducer and acceptor to create min edit distance spell checker with word frequencies
fstcompose fsts/EV_sorted.binfst fsts/W_sorted.binfst fsts/EVW.binfst


# 9.z draw

# make test acceptor and save it as test_W.fst
python3 scripts/make_acceptor_word_freq.py 'vocab/testwords_9.txt'>./fsts/test_W.fst

# compile model to bin and save it as test.binfst
fstcompile --isymbols=./vocab/words.syms --osymbols=./vocab/words.syms ./fsts/test_W.fst ./fsts/test_W.binfst


# draw the FST and save as png
fstdraw --isymbols=./vocab/words.syms --osymbols=./vocab/words.syms -portrait ./fsts/test_W.binfst | dot -Tpng > ./fsts/test_W.png

# make test acceptor and save it as test_V.fst
python3 scripts/make_acceptor.py 'vocab/testwords_9.syms'>./fsts/test_V.fst

# compile model to bin and save it as test.binfst
fstcompile --isymbols=./vocab/chars.syms --osymbols=./vocab/words.syms ./fsts/test_V.fst ./fsts/test_V.binfst

fstarcsort --sort_type=olabel fsts/test_V.binfst fsts/test_V.binfst

# sort inputs of acceptor
fstarcsort --sort_type=ilabel fsts/test_W.binfst fsts/test_W.binfst

# compose
fstcompose fsts/test_V.binfst fsts/test_W.binfst fsts/test_VW.binfst

# draw the FST and save as png
fstdraw --isymbols=./vocab/chars.syms --osymbols=./vocab/words.syms -portrait ./fsts/test_VW.binfst | dot -Tpng > ./fsts/test_VW.png

# remove temp files
rm fsts/test_V.fst
rm fsts/test_W.fst
rm fsts/test_V.binfst
rm fsts/test_W.binfst
rm fsts/test_VW.binfst
