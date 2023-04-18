#!/bin/bash

#command line arg
WORDS=${1}

# 5.a
# make acceptor and save it as V.fst
python3 scripts/make_acceptor.py $WORDS >./fsts/V.fst

# 5.d
# compile model to bin and save it as V.binfst
fstcompile --isymbols=./vocab/chars.syms --osymbols=./$WORDS ./fsts/V.fst ./fsts/V.binfst

# 5.b,c
# optimize model and save it as V_opt.binfst
fstrmepsilon fsts/V.binfst | fstdeterminize | fstminimize >fsts/V_opt.binfst
