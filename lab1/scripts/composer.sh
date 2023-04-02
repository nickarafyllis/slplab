#!/bin/bash

# 6a
# sort outputs of transducer
fstarcsort --sort_type=olabel fsts/L.binfst fsts/L_sort.binfst

# sort inputs of acceptor
fstarcsort --sort_type=ilabel fsts/V_opt.binfst fsts/V_sorted.binfst 

# compose transducer and acceptor to create min edit distance spell checker
fstcompose fsts/L_sort.binfst fsts/V_sorted.binfst fsts/S.binfst