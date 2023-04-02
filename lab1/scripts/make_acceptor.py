import sys

#read words
def read_first_column(file):
    f=open(file,"r")
    lines=f.readlines()
    words=[]
    for x in lines:
        #read only first column
        words.append(x.split()[0])
    f.close()
    return words
    
def make_acceptor(lexicon = 'vocab/words.syms'):
    words = read_first_column(lexicon)

    state = 0
    for i in words[1:]: # 1: to except <epsilon>
        letters = list(i)
        # make 0 initial state for every word
        print("0 %d %s %s" % (state+1, letters[0], i))  
        state+=1
        # for every intermediate letter go to next state 
        for j in letters[1:len(letters)]:
            print("%d %d %s <eps>" % (state, state+1, j))   
            state+=1
        # return to state 0 in last letter
        print("%d" % (state))
        
make_acceptor(sys.argv[1])