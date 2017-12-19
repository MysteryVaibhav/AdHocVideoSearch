import gensim
import queue as Q
import numpy as np
from PhraseVector import PhraseVector
from PreProcess import getSINconcepts, current_milli_time, data_dir, getProcessedConcepts

# Load Stanford's pre-trained glove model.
print("Loading model...")
path_to_trained_model = 'C:\\Users\\myste\\Downloads\\glove.6B\\glove.6B.100d.txt.word2vec'
model = gensim.models.KeyedVectors.load_word2vec_format(path_to_trained_model, binary=False)
print("Loading model... [OK]")

# Getting word embeddings for all the concepts
phraseVectors = []
#concepts = getSINconcepts()
concepts, raw = getProcessedConcepts(path_to_concept_file=data_dir + "AllConcepts.txt")
for concept in concepts:
    phraseVectors.append(PhraseVector(model, concept))

# top_k: Retrieves top k most similar concepts
top_k = 10;


def match(inputQuery):
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
    print("Found in {} milli secs".format(current_milli_time() - st))
    return arr
