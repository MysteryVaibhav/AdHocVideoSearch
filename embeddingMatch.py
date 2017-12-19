import queue as Q
import numpy as np
from PhraseVector import PhraseVector
from model import model, concept_ids, phraseVectors, semantic_concepts as concepts
from properties import top_k


def match(inputQuery):
    queryVector = PhraseVector(model, inputQuery)
    idx = 0;
    total_score = 0
    q = Q.PriorityQueue();
    for conceptVector in phraseVectors:
        score = queryVector.CosineSimilarity(conceptVector.vector)
        q.put((-1 * score, concepts[idx], concept_ids[idx]))
        total_score += score
        idx += 1

    idx = 0
    arr = []
    listOfConcepts = []
    conceptScores = []
    while (not q.empty()) and idx < top_k:
        elem = q.get()
        listOfConcepts.append(elem[1])
        conceptScores.append(-1 * elem[0] / total_score)
        arr.append(elem[2])
        idx += 1
    return arr, listOfConcepts, conceptScores
