#!/bin/bash

# write the fst file
python scripts/make_L.py > ./fsts/L.fst

# compile the fst to bin
fstcompile --isymbols=./vocab/chars.syms --osymbols=./vocab/chars.syms ./fsts/L.fst ./fsts/L.binfst

# draw the FST and save as png
fstdraw --isymbols=./vocab/chars.syms --osymbols=./vocab/chars.syms ./fsts/L.binfst | dot -Tpng > ./fsts/L.png
