#!/bin/bash

# sort outputs of transducer
fstarcsort --sort_type=olabel fsts/E11.binfst fsts/E11_sort.binfst

# sort inputs of acceptor
fstarcsort --sort_type=ilabel fsts/V_opt.binfst fsts/V_sorted.binfst

# compose transducer and acceptor to create min edit distance spell checker
fstcompose fsts/E11_sort.binfst fsts/V_sorted.binfst fsts/EV.binfst
