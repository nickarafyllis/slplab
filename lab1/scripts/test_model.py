from gensim.models import Word2Vec
import sys

# 12c
def test_model(path):

    model = Word2Vec.load(path)
    
    input_words = ['bible', 'book', 'bank', 'water']

    for input_word in input_words:
        
        # get most similar words
        most_similar = model.wv.most_similar(input_word, topn=5)

        # Print the results
        print(input_word)
        for word, score in most_similar:
            print(word, score)
            
test_model(sys.argv[1])