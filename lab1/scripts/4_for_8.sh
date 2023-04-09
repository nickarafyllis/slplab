#!/bin/bash

# 8d,e,st
# save edits
# count freqs
# write the fst file
python scripts/8.py > ./fsts/E.fst

# compile the fst to bin
fstcompile --isymbols=./vocab/chars.syms --osymbols=./vocab/chars.syms ./fsts/E.fst ./fsts/E.binfst
