import re
import gensim
import time
import queue as Q
import numpy as np
from PhraseVector import PhraseVector

# Load Stanford's pre-trained glove model.
print("Loading model...")
path_to_trained_model = 'C:\\Users\\myste\\Downloads\\GoogleNews-vectors-negative300.bin'
model = gensim.models.KeyedVectors.load_word2vec_format(path_to_trained_model, binary=True)
print("Loading model... [OK]")

# To calculate time
current_milli_time = lambda: int(round(time.time() * 1000))

data_dir = "C:\\Users\\myste\\Google Drive\\CMU\\Sem1\\Research\\Data\\AVS\\AVS_concept_detection\\AVS_concept_detection\\"
#concepts_sin = "concept_names_ImageNet.txt"
concepts_sin = "concept_names_SIN.txt"
file = open(data_dir+concepts_sin, "r", encoding='utf8')
# Getting all the concepts
concepts = []
print("Extracting SIN concepts....")
concepts = []
for line in file.readlines():
    line = line.split(":")[1].strip()
    for match in re.findall("_", line):
        line = line.replace("_", " ")
    concepts.append(line)
print("Extracting SIN concepts....[OK]")

# Getting word embeddings for all the concepts
phraseVectors = []
for concept in concepts:
    phraseVectors.append(PhraseVector(model, concept))

# top_k: Retrieves top k most similar concepts
top_k = 5;


def word2VecBestMatch(inputQuery):
    inputQuery = inputQuery.lower()
    inputQuery = inputQuery.replace("\n", "")
    print("Finding best match for " + inputQuery)
    st = current_milli_time()
    queryVector = PhraseVector(model, inputQuery)
    idx = 0;
    q = Q.PriorityQueue();
    for conceptVector in phraseVectors:
        score = queryVector.CosineSimilarity(conceptVector.vector)
        q.put((-1 * score, idx, concepts[idx]))
        idx += 1
    idx = 0
    print("Best {} matches using word vectors: ".format(top_k))

    arr = np.zeros(len(concepts))
    scores = 0
    while (not q.empty()) and idx < top_k:
        idx += 1;
        print(q.get());
        arr[q.get()[1]] = -1*q.get()[0]
        scores += arr[q.get()[1]]
    #for i in range(0, len(arr)):
     #   arr[i] /= scores
    print("Found in {} milli secs".format(current_milli_time() - st))
    return arr
