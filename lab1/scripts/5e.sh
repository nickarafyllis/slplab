#!/bin/bash

# 5.e draw

# make test acceptor and save it as test.fst
python3 scripts/make_acceptor.py 'vocab/testwords.syms'>./fsts/test.fst

# compile model to bin and save it as test.binfst
fstcompile --isymbols=./vocab/chars.syms --osymbols=./vocab/testwords.syms ./fsts/test.fst ./fsts/test.binfst

# draw the FST and save as png
fstdraw --isymbols=./vocab/chars.syms --osymbols=./vocab/testwords.syms ./fsts/test.binfst | dot -Tpng > ./fsts/test.png

# optimize model and save it as test.binfst
fstrmepsilon ./fsts/test.binfst ./fsts/test.binfst
fstdraw --isymbols=./vocab/chars.syms --osymbols=./vocab/testwords.syms ./fsts/test.binfst | dot -Tpng > ./fsts/testrmepsilon.png

fstdeterminize ./fsts/test.binfst ./fsts/test.binfst
fstdraw --isymbols=./vocab/chars.syms --osymbols=./vocab/testwords.syms ./fsts/test.binfst | dot -Tpng > ./fsts/testdeterminize.png

fstminimize ./fsts/test.binfst ./fsts/test.binfst
fstdraw --isymbols=./vocab/chars.syms --osymbols=./vocab/testwords.syms ./fsts/test.binfst | dot -Tpng > ./fsts/testminimize.png

# remove temp files
rm fsts/test.fst
rm fsts/test.binfst