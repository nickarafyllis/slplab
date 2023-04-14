from gensim.models import Word2Vec
import sys

word1, word2, word3 = sys.argv[1:4]

def analogy(word1, word2, word3):
    # Load the trained Word2Vec model
    model = Word2Vec.load("models/gutenberg_w2v.1000e.5w.model")

    # Find the word that is to "king" as "queen" is to what?
    try:
        result = model.wv.most_similar(positive=[word1, word3], negative=[word2], topn=1)
        print(result[0][0])
    except KeyError:
        print("Unable to find a suitable word that satisfies the analogy.")
        

analogy(word1, word2, word3)