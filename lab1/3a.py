def alphabet2index(alphabet = 'abcdefghijklmnopqrstuvwxyz'):

    f = open("vocab/chars.syms", "w+")
    f.write("<epsilon>\t" + str(0) + '\n')
    for i, letter in enumerate(alphabet, 1):
        f.write(letter + "\t " + str(i) + '\n')
    f.close()

alphabet2index()
