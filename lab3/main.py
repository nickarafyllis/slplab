import os
import warnings

from sklearn.exceptions import UndefinedMetricWarning
from sklearn.preprocessing import LabelEncoder
import torch
from torch.utils.data import DataLoader

from config import EMB_PATH
from dataloading import SentenceDataset
from models import BaselineDNN, MaxPoolingDNN, LSTM
from attention import SimpleSelfAttentionModel, MultiHeadAttentionModel
from training import train_dataset, eval_dataset, torch_train_val_split
from early_stopper import EarlyStopper
from utils.load_datasets import load_MR, load_Semeval2017A
from utils.load_embeddings import load_word_vectors

import numpy as np
from sklearn.metrics import f1_score, accuracy_score, recall_score
import matplotlib.pyplot as plt
import sys

warnings.filterwarnings("ignore", category=UndefinedMetricWarning)

########################################################
# Configuration
########################################################


# Download the embeddings of your choice
# for example http://nlp.stanford.edu/data/glove.6B.zip

# 1 - point to the pretrained embeddings file (must be in /embeddings folder)
EMBEDDINGS = os.path.join(EMB_PATH, "glove.6B.50d.txt")

# 2 - set the correct dimensionality of the embeddings
EMB_DIM = 50

EMB_TRAINABLE = False
BATCH_SIZE = 128
EPOCHS = 50
DATASET = "MR"  # options: "MR", "Semeval2017A"

# if your computer has a CUDA compatible gpu use it, otherwise use the cpu
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

########################################################
# Define PyTorch datasets and dataloaders
########################################################

# load word embeddings
print("loading word embeddings...")
word2idx, idx2word, embeddings = load_word_vectors(EMBEDDINGS, EMB_DIM)

# load the raw data
if DATASET == "Semeval2017A":
    X_train, y_train, X_test, y_test = load_Semeval2017A()
elif DATASET == "MR":
    X_train, y_train, X_test, y_test = load_MR()
else:
    raise ValueError("Invalid dataset")

# convert data labels from strings to integers
le = LabelEncoder()


y_train = le.fit_transform(y_train)  # EX1
y_test = le.fit_transform(y_test)  # EX1
n_classes = le.classes_.size  # EX1 - LabelEncoder.classes_.size

# Question 1
for i in range(10):
    print("Label: {}, Number: {}".format(le.inverse_transform([y_train[i]])[0], y_train[i]))



# Define our PyTorch-based Dataset
train_set = SentenceDataset(X_train, y_train, word2idx)
test_set = SentenceDataset(X_test, y_test, word2idx)

# Question 2
for i, ex in enumerate(train_set.data[:10]):
    print("Example {}: {}".format(i+1, ex))

# Make histogram of lengths of sentences
lengths = []
for ex in train_set.data:
    lengths.append(len(ex))

hist_of_lens = {}
for i in range(max(lengths)):
    hist_of_lens[i+1] = lengths.count(i+1)

print(hist_of_lens) # We choose 50 as a vector size that covers most of the sentences

# Question 3
for i, ex in enumerate(train_set.data[:5]):
    print('Example {0}: {1}\nTransformed example {0}: {2}'.format(i+1, ex, train_set[i]))



# EX7 - Define our PyTorch-based DataLoader
train_loader, val_loader = torch_train_val_split(
    train_set, batch_train=BATCH_SIZE, batch_eval=BATCH_SIZE
) #2.1
#train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)  # EX7
test_loader = DataLoader(train_set, batch_size=BATCH_SIZE)   # EX7

#############################################################################
# Model Definition (Model, Loss Function, Optimizer)
#############################################################################


model_name = sys.argv[1] # (BaselineDNN, MaxPoolingDNN, LSTM, LSTMbi)
if model_name == "LSTMbi":
    model = LSTM(output_size=n_classes,  # EX8
                        embeddings=embeddings,
                        trainable_emb=EMB_TRAINABLE, bidirectional=True)
else:
    model = eval(model_name)(output_size=n_classes,  # EX8
                    embeddings=embeddings,
                    trainable_emb=EMB_TRAINABLE)

# move the mode weight to cpu or gpu
model.to(DEVICE)
print(model)

# We optimize ONLY those parameters that are trainable (p.requires_grad==True)
if (n_classes == 3):
    criterion = torch.nn.CrossEntropyLoss()  # EX8
elif (n_classes == 2):
    criterion = torch.nn.BCEWithLogitsLoss()

parameters = filter(lambda p: p.requires_grad, model.parameters()) # EX8
optimizer = torch.optim.Adam(parameters) # EX8

#############################################################################
# Training Pipeline
#############################################################################
TRAIN_LOSS = []
TEST_LOSS = []


save_path = f'{DATASET}_{model.__class__.__name__}.pth'

# Stop if validation loss keeps increasing for 5 epochs
early_stopper = EarlyStopper(model, save_path, patience=5)

for epoch in range(1, EPOCHS + 1):
    # train the model for one epoch
    train_dataset(epoch, train_loader, model, criterion, optimizer, DATASET)

    # evaluate the performance of the model, on both data sets
    train_loss, (y_train_gold, y_train_pred) = eval_dataset(train_loader,
                                                            model,
                                                            criterion,DATASET)

    validation_loss, (y_validation_gold, y_validation_pred) = eval_dataset(val_loader,
                                                            model,
                                                            criterion,DATASET)

    # 2.1
    if model_name.startswith("LSTM") and early_stopper.early_stop(validation_loss):
        print("Early stopping triggered. Training stopped.")
        EPOCHS = epoch-1 # for plot
        break

    test_loss, (y_test_gold, y_test_pred) = eval_dataset(test_loader,
                                                         model,
                                                         criterion, DATASET)



    TRAIN_LOSS.append(train_loss)
    TEST_LOSS.append(test_loss)
    # compute metrics using sklearn functions

    print("Train loss:" , train_loss)
    print("Test loss:", test_loss)
    print("Train accuracy:" , accuracy_score(y_train_gold, y_train_pred))
    print("Test accuracy:" , accuracy_score(y_test_gold, y_test_pred))
    print("Train F1 score:", f1_score(y_train_gold, y_train_pred, average='macro'))
    print("Test F1 score:", f1_score(y_test_gold, y_test_pred, average='macro'))
    print("Train Recall:", recall_score(y_train_gold, y_train_pred, average='macro'))
    print("Test Recall:", recall_score(y_test_gold, y_test_pred, average='macro'))

plt.plot(range(1, EPOCHS + 1), TRAIN_LOSS, label='Training Loss')
plt.plot(range(1, EPOCHS + 1), TEST_LOSS, label='Test Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training and Test Loss')
plt.legend()
plt.show()
