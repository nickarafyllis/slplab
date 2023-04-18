import sys
import numpy as np

def make_acceptor_wf(lexicon = 'vocab/words.vocab.txt'):
    f=open(lexicon,"r")
    lines = [ln.strip().split() for ln in f.readlines()]

    f.close()

    total = sum([int(ln[1]) for ln in lines])

    state = 0
    for ln in lines:
        #ln[0] is a word, ln[1] is its frequency
        weight = -np.log(int(ln[1])/total)
        print("0 0 %s %s %.2f" % (ln[0], ln[0], weight))
    print("0")

make_acceptor_wf(sys.argv[1])
