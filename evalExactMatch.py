import numpy as np
from PreProcess import current_milli_time, getProcessedConcepts, data_dir

concepts, raw = getProcessedConcepts(path_to_concept_file=data_dir + "AllConcepts.txt")


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
        for word in inputQuery.split(" "):
            if concept_set.__contains__(word):
                score += 1
        arr[idx] = score
        if score > 0:
            foundMatches.append(concept)
            scores.append(score)
        idx += 1
    print("Found matches: {}".format(str(foundMatches)))
    print("Scores of the match: {}".format(str(scores)))
    print("Found in {} milli secs".format(current_milli_time() - st))
    return arr
