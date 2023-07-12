from torch.utils.data import Dataset
from tqdm import tqdm
import nltk
nltk.download('punkt')
from nltk.tokenize import word_tokenize
import numpy as np

# import TweetTokenizer() method from nltk
from nltk.tokenize import TweetTokenizer
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer


class SentenceDataset(Dataset):
    """
    Our custom PyTorch Dataset, for preparing strings of text (sentences)
    What we have to do is to implement the 2 abstract methods:

        - __len__(self): in order to let the DataLoader know the size
            of our dataset and to perform batching, shuffling and so on...

        - __getitem__(self, index): we have to return the properly
            processed data-item from our dataset with a given index
    """

    def __init__(self, X, y, word2idx):
        """
        In the initialization of the dataset we will have to assign the
        input values to the corresponding class attributes
        and preprocess the text samples

        -Store all meaningful arguments to the constructor here for debugging
         and for usage in other methods
        -Do most of the heavy-lifting like preprocessing the dataset here


        Args:
            X (list): List of training samples
            y (list): List of training labels
            word2idx (dict): a dictionary which maps words to indexes
        """

        # Tokenize text
        # self.data = [word_tokenize(x) for x in X]
        # self.labels = y
        # self.word2idx = word2idx

        self.data = []
        # Tokenize text
        Dataset = "MR"
        if Dataset == "MR":
            # heavy tokenization
            # Tokenization and preprocessing
            nltk.download('wordnet')
            nltk.download('stopwords')
            nltk.download('omw-1.4')
            stop_words = set(stopwords.words('english'))
            lemmatizer = WordNetLemmatizer()

            for sample in X:
                # Tokenize the sample
                tokens = nltk.word_tokenize(sample)

                # Apply preprocessing steps
                tokens = [lemmatizer.lemmatize(token.lower()) for token in tokens if token.isalnum()]
                tokens = [token for token in tokens if token not in stop_words]

                self.data.append(tokens)

            #self.data = [[w for w in x.split(" ") if len(w) > 0] for x in X]

        elif Dataset == "Semeval2017A":
            # Create an instance of the TweetTokenizer
            tokenizer = TweetTokenizer()
            # Tokenize the dataset
            self.data  = [tokenizer.tokenize(tweet) for tweet in X]

        self.labels = y
        self.word2idx = word2idx

        # EX2


    def __len__(self):
        """
        Must return the length of the dataset, so the dataloader can know
        how to split it into batches

        Returns:
            (int): the length of the dataset
        """

        return len(self.data)

    def __getitem__(self, index):
        """
        Returns the _transformed_ item from the dataset

        Args:
            index (int):

        Returns:
            (tuple):
                * example (ndarray): vector representation of a training example
                * label (int): the class label
                * length (int): the length (tokens) of the sentence

        Examples:
            For an `index` where:
            ::
                self.data[index] = ['this', 'is', 'really', 'simple']
                self.target[index] = "neutral"

            the function will have to return something like:
            ::
                example = [  533  3908  1387   649   0     0     0     0]
                label = 1
                length = 4
        """

        # EX3
        # Map tokens to numbers
        # word2idx = self.word2idx
        # example = np.array([word2idx[w] if w in word2idx else word2idx['<unk>'] for w in self.data[index]])

        # # We choose 50 as a vector size that covers most of the sentences
        # length = 50
        # if len(example) > 50:
        #     example = example[:50]

        # if len(example) < 50:
        #     length = len(example)
        #     example = np.concatenate((example,np.zeros(50-length)))

        # label = self.labels[index]


        # return example, label, length

        sentence = self.data[index]

        # Encode the sentence using word-to-id mapping
        encoded_sentence = [self.word2idx.get(word, self.word2idx["<unk>"]) for word in sentence]

        # Pad or truncate the sentence to a fixed length
        length = 42  # Choose a suitable maximum length
        if len(encoded_sentence) < length:
            encoded_sentence += [0] * (length - len(encoded_sentence))
            length = len(encoded_sentence)
        else:
            encoded_sentence = encoded_sentence[:length]

        length

        return np.array(encoded_sentence), self.labels[index], length
