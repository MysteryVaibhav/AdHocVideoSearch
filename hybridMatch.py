import queue as Q
from model import semantic_concepts as concepts, stripCommonPhrases, map_word_synsets
from model import current_milli_time, getNounsAndVerbs, getSynonyms, wn, tfidf, np, concept_ids
from properties import top_k
from embeddingMatch import match


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
            minScore = 100000
            synsets = []
            if cw in map_word_synsets:
                synsets = map_word_synsets[cw]
            else:
                synsets = wn.synsets(cw)
            for synset in synsets:
                depth = synset.max_depth()
                minScore = min(depth,minScore) + 1
            if minScore == 100000:
                minScore = 0
            tfIdf = tfidf(cw)
            maxScore = max(maxScore, tfIdf*minScore)
        score = (len(commonWords)/len(conceptWords)) * maxScore
    return score


def bestMatch(sentence):
    sentence = stripCommonPhrases(sentence.replace("\n", "").lower())
    st = current_milli_time()
    nouns, verbs = getNounsAndVerbs(sentence)

    phrases = nouns.copy()
    phrases.update(verbs)
    phrases = list(phrases)
    biPhrases = phrases.copy()
    # for noun in nouns:
    #     for verb in verbs:
    #         biPhrases.append(noun + " " + verb)
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
    arr = []
    q = Q.PriorityQueue()
    idx = 0
    total_score = 0
    for score in conceptVector:
        if score != 0:
            q.put((-1*score, concepts[idx], concept_ids[idx]))
            total_score += score
        idx += 1
    idx = 0
    if total_score == 0:
        print("No matches found using hybrid approach !!")
        print("Using glove embeddings now !!")
        arr, listOfConcepts, conceptScores = match(sentence)
    else:
        while (not q.empty()) and idx < top_k:
            elem = q.get()
            listOfConcepts.append(elem[1])
            conceptScores.append(-1*elem[0]/total_score)
            arr.append(elem[2])
            idx += 1
    print("Computed in {} milli secs".format(current_milli_time() - st))
    print(sentence + " :: " + str(listOfConcepts))
    print(str(conceptScores))
    print(arr)
    return arr
