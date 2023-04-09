from helpers import run_cmd
from util import INFINITY
from tqdm import tqdm
import numpy as np

#d)
def save_edits():
    with open("data/wiki.txt", "r") as fd1, open("data/edits.txt", "w") as fd2:
        lines = [ln.strip().split("\t") for ln in fd1.readlines()]

        for ln in tqdm(lines):
            #ln[0] is wrong, ln[1] is correct
            edits = run_cmd("bash scripts/word_edits.sh {0} {1}".format(ln[0],ln[1]))
            if len(edits) < 10:
                # if len > 10, command failed and there is an error message
                fd2.write(edits)

        fd1.close()
        fd2.close()

save_edits()

#e)
def count_frequencies():
    dict = {}
    with open("data/edits.txt", "r") as fd:
        lines = [ln.strip().split() for ln in fd.readlines()]

        for ln in tqdm(lines):
            #ln[0] is source, ln[1] is destination
            key = tuple((ln[0],ln[1]))
            if key not in dict:
                dict[key] = 1
            else:
                dict[key] += 1

        fd.close()

    return dict

frequencies = count_frequencies()

#st)
alphabet = 'abcdefghijklmnopqrstuvwxyz'

total = sum(frequencies.values())

#new weights depending on frequencies
weights = {k:-np.log(v/total) for k,v in frequencies.items()}
# No edit
for l in alphabet:
    print ("0 0 %s %s %.1f" % (l, l, 0))

# Deletes: input character, output epsilon
for l in alphabet:
    key = (l,"<eps>")
    if key in weights:
        weight = weights[key]
    else:
        weight = INFINITY
    print ("0 0 %s <eps> %.1f" % (l, weight))

# Insertions: input epsilon, output character
for l in alphabet:
    key = ("<eps>",l)
    if key in weights:
        weight = weights[key]
    else:
        weight = INFINITY
    print ("0 0 <eps> %s %.1f" % (l, weight))

# Substitutions: input one character, output another
for l in alphabet:
    for r in alphabet:
        if l is not r:
            key = (l,r)
            if key in weights:
                weight = weights[key]
            else:
                weight = INFINITY
            print ("0 0 %s %s %.1f" % (l, r, weight))

# Final state
print (0)
