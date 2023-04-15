#!/bin/bash

python scripts/11.py > ./fsts/E11.fst

# compile the fst to bin
fstcompile --isymbols=./vocab/chars.syms --osymbols=./vocab/chars.syms ./fsts/E11.fst ./fsts/E11.binfst
