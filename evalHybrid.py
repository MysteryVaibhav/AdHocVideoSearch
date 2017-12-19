import numpy as np
from nltk.corpus import wordnet as wn
from tfIdf import idf, idf1
from nlpTasks import getSynonyms, getNounsAndVerbs, getSimilarWords
from PreProcess import current_milli_time, getSINconcepts, getProcessedConcepts, data_dir
from sematic_role_labelling import getRelevantSubQueries

#concepts = getSINconcepts()
concepts, raw = getProcessedConcepts(path_to_concept_file=data_dir + "AllConcepts.txt") #"AllConcepts.txt")


def computeSimilarity(queryPhrase, concept):
    score = 0
    queryWords = set(queryPhrase.split(" "))
    conceptWords = set(concept.split(" "))
    conceptSynonyms = set()
    for cw in conceptWords:
        conceptSynonyms.update(getSynonyms(cw, wn.NOUN))
        conceptSynonyms.update(getSynonyms(cw, wn.VERB))
        #conceptSynonyms.update(getSimilarWords(cw))
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
            tfIdf = idf1(cw)
            maxScore = max(maxScore, tfIdf*minScore)
        score = (len(commonWords)/len(conceptWords)) * maxScore #+ 10*len(commonWords)
    return score


def match(sentence):
    #print(sentence)
    st = current_milli_time()
    # nouns, verbs = getNounsAndVerbs(sentence)
    # nouns.discard("person")
    # nouns.discard("people")
    # nouns.discard("man")
    # nouns.discard("woman")
    # phrases = nouns.copy()
    # phrases.update(verbs)
    # phrases = list(phrases)
    # biPhrases = phrases.copy()
    # for phrase in phrases:
    #     for p in phrases:
    #         if p != phrase:
    #             biPhrases.append(p + " " + phrase)
    # phrases = biPhrases

    phrases = list(getRelevantSubQueries(sentence))
    phrases = [word.lower() for word in phrases]
    numPhrases = len(phrases)
    numConcepts = len(concepts)

    #Similarity Matrix
    S = np.zeros((numConcepts, numPhrases))

    for i in range(0, numConcepts):
        for j in range(0, numPhrases):
            ss = computeSimilarity(phrases[j], concepts[i])
            S[i][j] = ss
    # sMax = S.max(0)
    # for j in range(0, numPhrases):
    #     for i in range(0, numConcepts):
    #         if sMax[j] == S[i][j]:
    #             S[i][j] = sMax[j]
    #         else:
    #             S[i][j] = 0
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
    print("Computed in {} milli secs".format(current_milli_time() - st))
    print(sentence + " :: " + str(listOfConcepts))
    print(str(conceptScores))
    return arr
