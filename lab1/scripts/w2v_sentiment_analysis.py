import glob
import os
import re
import sys

import numpy as np
import sklearn
from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

from gensim.models import Word2Vec

from gensim.models import KeyedVectors
NUM_W2V_TO_LOAD = 1000000

SCRIPT_DIRECTORY = os.path.realpath(__file__)

#data_dir = os.path.join(SCRIPT_DIRECTORY, "../data/aclImdb/")
data_dir = "data/aclImdb/"
train_dir = os.path.join(data_dir, "train")
test_dir = os.path.join(data_dir, "test")
pos_train_dir = os.path.join(train_dir, "pos")
neg_train_dir = os.path.join(train_dir, "neg")
pos_test_dir = os.path.join(test_dir, "pos")
neg_test_dir = os.path.join(test_dir, "neg")

# For memory limitations. These parameters fit in 8GB of RAM.
# If you have 16G of RAM you can experiment with the full dataset / W2V
MAX_NUM_SAMPLES = 5000
# Load first 1M word embeddings. This works because GoogleNews are roughly
# sorted from most frequent to least frequent.
# It may yield much worse results for other embeddings corpora
NUM_W2V_TO_LOAD = 1000000


SEED = 42

# Fix numpy random seed for reproducibility
np.random.seed(SEED)


def strip_punctuation(s):
    return re.sub(r"[^a-zA-Z\s]", " ", s)


def preprocess(s):
    return re.sub("\s+", " ", strip_punctuation(s).lower())


def tokenize(s):
    return s.split(" ")


def preproc_tok(s):
    return tokenize(preprocess(s))


def read_samples(folder, preprocess=lambda x: x):
    samples = glob.iglob(os.path.join(folder, "*.txt"))
    data = []

    for i, sample in enumerate(samples):
        if MAX_NUM_SAMPLES > 0 and i == MAX_NUM_SAMPLES:
            break
        with open(sample, "r") as fd:
            x = [preprocess(l) for l in fd][0]
            data.append(x)

    return data


def create_corpus(pos, neg):
    corpus = np.array(pos + neg, dtype='object')
    y = np.array([1 for _ in pos] + [0 for _ in neg])
    indices = np.arange(y.shape[0])
    np.random.shuffle(indices)

    return list(corpus[indices]), list(y[indices])


def extract_nbow(corpus, google):
    """Extract neural bag of words representations"""
    
    # Compute BoW representation with average of word embeddings
    
    if (google==1):
        # Load the trained Word2Vec model
        word_vectors = KeyedVectors.load_word2vec_format('./models/GoogleNews-vectors-negative300.bin', binary=True, limit=NUM_W2V_TO_LOAD)
    
    else:
        # Load the trained Word2Vec model
        model = Word2Vec.load("models/gutenberg_w2v.1000e.5w.model")
        
        # Obtain word embeddings
        word_vectors = model.wv
    
    bow_corpus=[]
    for doc in corpus:
        word_embeddings = []
        for word in doc:
            if word in word_vectors:
                word_embeddings.append(word_vectors[word])
        if not word_embeddings:
            bow_corpus.append(np.zeros(word_vectors.vector_size))
        else:
            bow_corpus.append(np.mean(word_embeddings, axis=0))

    return bow_corpus
    raise NotImplementedError("Implement nbow extractor")


def train_sentiment_analysis(train_corpus, train_labels):
    """Train a sentiment analysis classifier using NBOW + Logistic regression"""
    
    # Define the clasifier
    clf = LogisticRegression()
    # Train the model
    clf.fit(train_corpus, train_labels)
    return clf

    raise NotImplementedError("Implement sentiment analysis training")


def evaluate_sentiment_analysis(classifier, test_corpus, test_labels):
    """Evaluate classifier in the test corpus and report accuracy"""
    
    return accuracy_score(test_labels, classifier.predict(test_corpus))
    
    raise NotImplementedError("Implement sentiment analysis evaluation")


if __name__ == "__main__":
    # TODO: read Imdb corpus
    corpus, labels = create_corpus(read_samples(pos_train_dir, preproc_tok), read_samples(neg_train_dir, preproc_tok))
    google=sys.argv[1]
    if (google=='google'):
        nbow_corpus = extract_nbow(corpus,1)
    else:
        nbow_corpus = extract_nbow(corpus,0)
    (
        train_corpus,
        test_corpus,
        train_labels,
        test_labels,
    ) = train_test_split(nbow_corpus, labels)

    # TODO: train / evaluate and report accuracy
    
    trained_classifier = train_sentiment_analysis(train_corpus, train_labels)
    
    accuracy = evaluate_sentiment_analysis(trained_classifier, test_corpus, test_labels)
    
    print("Accuracy score: ", accuracy)