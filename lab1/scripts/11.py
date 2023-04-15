import numpy as np


def count_frequencies():
    dict = {}
    with open("data/edits.txt", "r") as fd:
        lines = [ln.strip().split() for ln in fd.readlines()]

        for ln in lines:
            #ln[0] is source, ln[1] is destination
            key = tuple((ln[0],ln[1]))
            if key not in dict:
                dict[key] = 1
            else:
                dict[key] += 1

        fd.close()

    return dict

frequencies = count_frequencies()
N = sum(frequencies.values()) # total number of edits

alphabet = 'abcdefghijklmnopqrstuvwxyz'
num = len(alphabet) #26
V = 2*num + num*(num-1) # number of unique edits 26*27

# Add-1 smoothing
frequencies = {k:(v+1)*N/(N+V) for k,v in frequencies.items()}

#new weights depending on frequencies
weights = {k:-np.log(v/N) for k,v in frequencies.items()}
# No edit
for l in alphabet:
    print ("0 0 %s %s %.1f" % (l, l, 0))

# Deletes: input character, output epsilon
for l in alphabet:
    key = (l,"<eps>")
    if key in weights:
        weight = weights[key]
    else:
        weight = -np.log(1/(N+V))
    print ("0 0 %s <eps> %.1f" % (l, weight))

# Insertions: input epsilon, output character
for l in alphabet:
    key = ("<eps>",l)
    if key in weights:
        weight = weights[key]
    else:
        weight = -np.log(1/(N+V))
    print ("0 0 <eps> %s %.1f" % (l, weight))

# Substitutions: input one character, output another
for l in alphabet:
    for r in alphabet:
        if l is not r:
            key = (l,r)
            if key in weights:
                weight = weights[key]
            else:
                weight = -np.log(1/(N+V))
            print ("0 0 %s %s %.1f" % (l, r, weight))

# Final state
print (0)
