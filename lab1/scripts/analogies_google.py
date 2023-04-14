import sys
from gensim.models import KeyedVectors
NUM_W2V_TO_LOAD = 1000000

word1, word2, word3 = sys.argv[1:4]

def analogy(word1, word2, word3):
    # Load the trained Word2Vec model
    model = KeyedVectors.load_word2vec_format('./models/GoogleNews-vectors-negative300.bin', binary=True, limit=NUM_W2V_TO_LOAD)

    # Find the word that is to "king" as "queen" is to what?
    try:
        result = model.most_similar(positive=[word1, word3], negative=[word2], topn=1)
        print(result[0][0])
    except KeyError:
        print("Unable to find a suitable word that satisfies the analogy.")

analogy(word1, word2, word3)