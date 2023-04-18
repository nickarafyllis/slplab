from gensim.models import Word2Vec

# Load the trained Word2Vec model
model = Word2Vec.load("models/gutenberg_w2v.1000e.5w.model")

# Obtain word embeddings
word_vectors = model.wv

# Save the word embeddings into "embeddings.tsv"
with open("models/embeddings.tsv", "w") as f:
    for word in word_vectors.index_to_key:
        embedding = "\t".join(str(x) for x in word_vectors[word])
        f.write(f"{embedding}\n")
        
# Save the corresponding words into "metadata.tsv"
with open("models/metadata.tsv", "w") as f:
    # Write each word to the file
    for word in word_vectors.index_to_key:
        f.write(f"{word}\n")