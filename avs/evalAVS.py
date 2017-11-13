import queue as Q
import numpy as np

data_dir = "C:\\Users\\myste\\Google Drive\\CMU\\Sem1\\Research\\Data\\AVS\\AVS_concept_detection\\AVS_concept_detection\\"

# Loading video scores for SIN concept vectors
print("Extracting SIN concept scores....")
scores_sin = "try.txt"#"scores_SIN_svm.txt"
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
        q.put((-1*conceptVector.CosineSimilarity(score[2:]), int(score[0]), int(score[1])))
    idx = 0
    while (not q.empty()) and idx < top_k:
        vid = q.get()
        file.write("{} 0 shot{}_{}".format(qId,vid[1],vid[2]))
        idx += 1
