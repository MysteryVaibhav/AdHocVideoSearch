import math
import numpy as np
import queue as Q
from PreProcess import getQueries, data_dir, allConceptScoresInVideos, getQueriesRaw, conceptScoresInVideos
#from evalExactMatch import match
#from evalHybrid import match
#from evalGlove import match
#from evalWord2vec import match
#from evalWordNetWup import match
#from evalWordNetLch import match
#from hybrid1 import match
#from evalEsa import match
#from evalMyModel import match
from linguisticApproach import match

top_k = 1000


def matchVideos(qId, conceptVector, file):
    q = Q.PriorityQueue()
    for score in scores:
        q.put((-1*CosineSimilarity(conceptVector, score[2:]), int(score[0]), int(score[1])))
        #q.put((-1 * histogramSimilarity(conceptVector, score[2:]), int(score[0]), int(score[1])))
    idx = 0
    while (not q.empty()) and idx < top_k:
        vid = q.get()
        file.write("1{}0 shot{}_{} {} {} INF\n".format(qId, vid[1], vid[2], idx+1, 9999-idx))
        idx += 1


def CosineSimilarity(vector, otherConceptVec):
    cosine_similarity = np.dot(vector, otherConceptVec) / (
        np.linalg.norm(vector) * np.linalg.norm(otherConceptVec))
    try:
        if math.isnan(cosine_similarity):
            cosine_similarity = 0
    except:
        cosine_similarity = 0
    return cosine_similarity


def histogramSimilarity(vec1, vec2):
    score = np.sum(np.minimum(vec1,vec2))
    return score


if __name__ == '__main__':
    # Loading the queries
    #queries = getQueries()
    queries = getQueriesRaw()
    # Loading video scores for SIN concept vectors
    #scores = allConceptScoresInVideos()
    scores = conceptScoresInVideos()
    # Opening file to write the results
    trec = "trec_sin_imageNet.txt"
    file = open(data_dir + trec, "w", encoding="utf8")
    # Generating concept vectors for the query
    for query in queries:
        bestMatchVector = match(query[1])
        matchVideos(query[0], bestMatchVector, file)
    file.close()
