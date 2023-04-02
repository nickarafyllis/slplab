alphabet = 'abcdefghijklmnopqrstuvwxyz' #'abc' #for drawing fst at 4z
weight = {
    "delete": 1.0,
    "insert": 1.0,
    "sub": 1.0
}

# No edit
for l in alphabet:
    print ("0 0 %s %s %.1f" % (l, l, 0))

# Deletes: input character, output epsilon
for l in alphabet:
    print ("0 0 %s <eps> %.1f" % (l, weight["delete"]))

# Insertions: input epsilon, output character
for l in alphabet:
    print ("0 0 <eps> %s %.1f" % (l, weight["insert"]))

# Substitutions: input one character, output another
for l in alphabet:
    for r in alphabet:
        if l is not r:
            print ("0 0 %s %s %.1f" % (l, r, weight["sub"]))

# Final state
print (0)
