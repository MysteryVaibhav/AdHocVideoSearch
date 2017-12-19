import numpy as np
import json
import math
from PreProcess import current_milli_time, getProcessedConcepts, data_dir

concepts, raw = getProcessedConcepts(path_to_concept_file=data_dir + "AllConcepts.txt")
with open(data_dir + "captions_train2014.json", "r", encoding="utf-8") as f:
    dataStore = json.load(f)

#annotations -> caption
#Building vocabalury
VOCAB = {}
for annotation in dataStore['annotations']:
    caption = annotation['caption']
    caption = caption.replace(",", "").lower()
    for word in set(caption.split(" ")):
        if word not in VOCAB:
            VOCAB[word] = 1
        else:
            VOCAB[word] += 1


def idf(word):
    # For rare words
    if word not in VOCAB:
        return math.log(len(VOCAB))

    return math.log(len(VOCAB)/VOCAB[word])


def match(inputQuery):
    print("Finding best match for " + inputQuery)
    st = current_milli_time()
    idx = 0;
    arr = np.zeros(len(concepts))
    foundMatches = []
    scores = []
    for concept in concepts:
        score = 0
        concept_set = set(concept.split(" "))
        count = 0
        for word in inputQuery.split(" "):
            if "person" in word or "man" in word or "people" in word:
                continue
            if concept_set.__contains__(word):
                count += 1
                score += idf(word)
        arr[idx] = score
        if score > 0:
            foundMatches.append(concept)
            scores.append(score*count)
        idx += 1
    print("Found matches: {}".format(str(foundMatches)))
    print("Scores of the match: {}".format(str(scores)))
    print("Found in {} milli secs".format(current_milli_time() - st))
    return arr
