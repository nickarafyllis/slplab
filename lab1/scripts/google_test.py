from gensim.models import KeyedVectors
NUM_W2V_TO_LOAD = 1000000

model = KeyedVectors.load_word2vec_format('./models/GoogleNews-vectors-negative300.bin', binary=True, limit=NUM_W2V_TO_LOAD)

input_words = ['bible', 'book', 'bank', 'water']

for input_word in input_words:
    
    # get most similar words
    most_similar = model.most_similar(input_word, topn=5)

    # Print the results
    print(input_word)
    for word, score in most_similar:
        print(word, score)