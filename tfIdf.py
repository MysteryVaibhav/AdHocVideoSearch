import numpy as np
import math
import json
from nltk.corpus import reuters
from nltk.corpus import stopwords
from nltk import word_tokenize
from string import punctuation
from PreProcess import data_dir

stop_words = stopwords.words('english') + list(punctuation)


def tokenize(text):
    words = word_tokenize(text)
    words = [w.lower() for w in words]
    return [w for w in words if w not in stop_words and not w.isdigit()]

print("Building vocabulary to compute tf_idf score...")
# build the vocabulary in one pass
vocabulary = set()
for file_id in reuters.fileids():
    words = tokenize(reuters.raw(file_id))
    vocabulary.update(words)

vocabulary = list(vocabulary)
word_index = {w: idx for idx, w in enumerate(vocabulary)}

VOCABULARY_SIZE = len(vocabulary)
DOCUMENTS_COUNT = len(reuters.fileids())

word_idf = np.zeros(VOCABULARY_SIZE)
for file_id in reuters.fileids():
    words = set(tokenize(reuters.raw(file_id)))
    indexes = [word_index[word] for word in words]
    word_idf[indexes] += 1.0

word_idf = np.log(DOCUMENTS_COUNT / (1 + word_idf).astype(float))
#print(word_idf[word_index['deliberations']])  # 7.49443021503
#print(word_idf[word_index['committee']])  # 3.61286641709
print("Building vocabulary to compute tf_idf score... [OK]")


with open(data_dir + "captions_train2014.json", "r", encoding="utf-8") as f:
    dataStore = json.load(f)

#annotations -> caption
#Building vocabalury
VOCAB = {}
no = 0
for annotation in dataStore['annotations']:
    no += 1
    caption = annotation['caption']
    caption = caption.replace(",", "").lower()
    for word in set(caption.split(" ")):
        if word not in VOCAB:
            VOCAB[word] = 1
        else:
            VOCAB[word] += 1

print("Number of captions: " + str(no) + " :: Size of vocabulary: " + str(len(VOCAB)))

def idf1(word):
    # For rare words
    if word not in VOCAB:
        return math.log(len(VOCAB))

    return math.log(len(VOCAB)/VOCAB[word])


def idf(word):
    score = 0.1
    if word in word_index:
        score = word_idf[word_index[word]]
    return score