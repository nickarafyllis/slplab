from transformers import pipeline
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm
from utils.load_datasets import load_MR, load_Semeval2017A
from training import get_metrics_report



LABELS_MAPPING = {
    'siebert/sentiment-roberta-large-english': {
        'POSITIVE': 'positive',
        'NEGATIVE': 'negative',
    },
    'cardiffnlp/twitter-roberta-base-sentiment': {
        'LABEL_0': 'negative',
        'LABEL_1': 'neutral',
        'LABEL_2': 'positive',
    },
    'lxyuan/distilbert-base-multilingual-cased-sentiments-student': {
        'negative': 'negative',
        'neutral': 'neutral',
        'positive': 'positive',
    },
    'textattack/bert-base-uncased-imdb': {
        'LABEL_0': 'negative',
        'LABEL_1': 'positive',
    },
    'textattack/roberta-base-imdb': {
        'LABEL_0': 'negative',
        'LABEL_1': 'positive',
    },
    'finiteautomata/bertweet-base-sentiment-analysis': {
        'NEG': 'negative',
        'NEU': 'neutral',
        'POS': 'positive',
    },
}

PRETRAINED_MODELS_MR = [
    'siebert/sentiment-roberta-large-english',
    'textattack/bert-base-uncased-imdb',
    'textattack/roberta-base-imdb'
]
PRETRAINED_MODELS_SEMEVAL = [
    'cardiffnlp/twitter-roberta-base-sentiment',
    'lxyuan/distilbert-base-multilingual-cased-sentiments-student',
    'finiteautomata/bertweet-base-sentiment-analysis'
]

if __name__ == '__main__':

    for DATASET in ("MR", "Semeval2017A"):
        if DATASET == "Semeval2017A":
            PRETRAINED_MODELS = PRETRAINED_MODELS_SEMEVAL
            X_train, y_train, X_test, y_test = load_Semeval2017A()
        elif DATASET == "MR":
            PRETRAINED_MODELS = PRETRAINED_MODELS_MR
            X_train, y_train, X_test, y_test = load_MR()

        # encode labels
        le = LabelEncoder()
        le.fit(list(set(y_train)))
        y_train = le.transform(y_train)
        y_test = le.transform(y_test)
        n_classes = len(list(le.classes_))

        for PRETRAINED_MODEL in PRETRAINED_MODELS:
            # define a proper pipeline
            sentiment_pipeline = pipeline("sentiment-analysis", model=PRETRAINED_MODEL)

            y_pred = []
            for x in tqdm(X_test):
                label = sentiment_pipeline(x)[0]['label']
                y_pred.append(LABELS_MAPPING[PRETRAINED_MODEL][label])

            y_pred = le.transform(y_pred)
            print(f'\nDataset: {DATASET}\nPre-Trained model: {PRETRAINED_MODEL}\nTest set evaluation\n{get_metrics_report([y_test], [y_pred])}')
