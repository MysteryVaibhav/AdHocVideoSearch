import math
import numpy as np
import queue as Q
from zeroShot import getConcepts
#from glove import gloveBestMatch
#from word2vec import word2VecBestMatch

data_dir = "C:\\Users\\myste\\Google Drive\\CMU\\Sem1\\Research\\Data\\AVS\\AVS_concept_detection\\AVS_concept_detection\\"
query_file = "queries.txt"

# Loading the queries
file = open(data_dir+query_file, "r", encoding='utf8')
queries = []

for line in file.readlines():
    lines = line.split(":")
    queries.append((lines[0], lines[1].replace("Find shots of ", "")))

# Loading video scores for SIN concept vectors
print("Extracting SIN concept scores....")
scores_sin = "scores_ImageNet.txt"
#scores_sin = "scores_SIN_svm.txt"
file = open(data_dir+scores_sin, "r", encoding="utf8")
scores = []
for line in file.readlines():
    scores.append(np.array(list(map(float, line.split(" ")))))
print("Extracting SIN concept scores....[OK]")

trec = "trec.txt"
file = open(data_dir+trec, "w", encoding="utf8")
top_k = 1000


def bestMatch(qId, conceptVector):
    q = Q.PriorityQueue()
    for score in scores:
        q.put((-1*CosineSimilarity(conceptVector, score[2:]), int(score[0]), int(score[1])))
    idx = 0
    while (not q.empty()) and idx < top_k:
        vid = q.get()
        file.write("1{} 0 shot{}_{}\n".format(qId, vid[1], vid[2]))
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


# Generating concept vectors for the query
for query in queries:
    bestMatchVector = getConcepts(query[1])
    bestMatch(query[0], bestMatchVector)

file.close()