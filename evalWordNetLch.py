import queue as Q
import numpy as np
from nltk.corpus import wordnet as wn
from PreProcess import current_milli_time, getProcessedConcepts, data_dir

concepts, raw = getProcessedConcepts(path_to_concept_file=data_dir + "AllConcepts.txt")
# Filling up synsets for each concept
synsets = []
for line in concepts:
    sublist = []
    for word in line.split(" "):
        for synset in wn.synsets(word):
            sublist.append(synset)
            break
    synsets.append(sublist)

# top_k: Retrieves top k most similar concepts
top_k = 10;


def match(query):
    print("Finding best match for " + query)
    st = current_milli_time()
    querySyns = []
    for queryWord in query.split(" "):
        for querySyn in wn.synsets(queryWord):
            querySyns.append(querySyn)
            break;

    # Finding similar concept
    maxScoreWup = 0
    maxIdxWup = 0
    idx = 0
    q = Q.PriorityQueue();
    for synList in synsets:
        scoreLch = 0
        noMatchLch = 0
        for querySyn in querySyns:
            for syn in synList:
                indScore = 0
                if querySyn._pos == syn._pos:
                    indScore = wn.lch_similarity(querySyn, syn)
                if indScore is not None:
                    scoreLch += indScore
                else:
                    noMatchLch += 1
        if len(querySyns) * len(synList) - noMatchLch > 0:
            scoreLch /= len(querySyns) * len(synList) - noMatchLch
            q.put((-1 * scoreLch, idx, concepts[idx]))
        idx += 1

    print("Best {} matches using wordnet lch similarity: ".format(top_k))
    arr = np.zeros(len(concepts))
    idx = 0
    while (not q.empty()) and idx < top_k:
        idx += 1;
        print(q.get());
        arr[q.get()[1]] = -1 * q.get()[0]
    print("Found in {} milli secs".format(current_milli_time() - st))

    return arr
