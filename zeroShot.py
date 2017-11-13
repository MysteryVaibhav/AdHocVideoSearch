import numpy as np
import time
import re
from nltk.corpus import wordnet as wn
from tfIdf import idf
from nlpTasks import getSynonyms, getVerbs, getNouns

# To compute time
current_millis_time = lambda: int(round(time.time() * 1000))
# Getting concepts extracted from videos
data_dir = "C:\\Users\\myste\\Google Drive\\CMU\\Sem1\\Research\\Data\\AVS\\AVS_concept_detection\\AVS_concept_detection\\"
concepts_sin = "concept_names_ImageNet.txt"
#concepts_sin = "concept_names_SIN.txt"
file = open(data_dir+concepts_sin, "r", encoding='utf8')
# Getting all the concepts
print("Extracting SIN concepts....")
concepts = []
for line in file.readlines():
    line = line.split(":")[1].strip()
    line = line.replace("\n", "")
    for match in re.findall("_", line):
        line = line.replace("_", " ")
    concepts.append(line)
print("Extracting SIN concepts....[OK]")


def computeSimilarity(queryPhrase, concept):
    score = 0
    queryWords = set(queryPhrase.split(" "))
    conceptWords = set(concept.split(" "))
    conceptSynonyms = set()
    for cw in conceptWords:
        conceptSynonyms.update(getSynonyms(cw, wn.NOUN))
        conceptSynonyms.update(getSynonyms(cw, wn.VERB))
    commonWords = queryWords.intersection(conceptWords)
    commonWords.update(queryWords.intersection(conceptSynonyms))
    if len(commonWords) > 0:
        maxScore = 0
        for cw in commonWords:
            minScore = 100000   #S_u
            for synset in wn.synsets(cw):
                depth = synset.max_depth()
                minScore = min(depth,minScore) + 1
            if minScore == 100000:
                minScore = 0
            tfIdf = idf(cw)
            maxScore = max(maxScore, tfIdf*minScore)
        score = (len(commonWords)/len(conceptWords)) * maxScore #+ 10*len(commonWords)
    return score


def getConcepts(sentence):
    sentence = sentence.replace("\n", "")
    print(sentence)
    st = current_millis_time()
    nouns = getNouns(sentence)
    verbs = getVerbs(sentence)

    phrases = nouns.copy()
    phrases.update(verbs)
    phrases = list(phrases)
    biPhrases = phrases.copy()
    for phrase in phrases:
        for p in phrases:
            if p != phrase:
                biPhrases.append(p + " " + phrase)
    phrases = biPhrases
    numPhrases = len(phrases)
    numConcepts = len(concepts)

    #Similarity Matrix
    S = np.zeros((numConcepts, numPhrases))

    for i in range(0, numConcepts):
        for j in range(0, numPhrases):
            ss = computeSimilarity(phrases[j], concepts[i])
            S[i][j] = ss
    sMax = S.max(0)
    for j in range(0, numPhrases):
        for i in range(0, numConcepts):
            if sMax[j] == S[i][j]:
                S[i][j] = sMax[j]
            else:
                S[i][j] = 0
    conceptVector = S.max(1)
    i = 0
    listOfConcepts = []
    conceptScores = []
    arr = np.zeros(len(concepts))
    for line in concepts:
        if conceptVector[i] != 0:
            arr[i] = conceptVector[i]
            listOfConcepts.append(line)
            conceptScores.append(conceptVector[i])
        i = i + 1
    print("Computed in {} milli secs".format(current_millis_time() - st))
    print(str(listOfConcepts))
    print(str(conceptScores))
    return arr
