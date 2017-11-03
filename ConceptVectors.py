import en_core_web_sm
from spacy.symbols import VERB
import numpy as np
import re
from nltk.corpus import wordnet as wn
from tfIdf import idf
import time
from nltk.stem import WordNetLemmatizer

current_millis_time = lambda: int(round(time.time() * 1000))

lemmatizer = WordNetLemmatizer()
print("Loading NLP ...")
nlp = en_core_web_sm.load()
print("Loading NLP ... [OK]")

print("Loading all concepts ...")
file = open("C:\\Users\\myste\\Google Drive\\CMU\\Sem1\\Research\\Concepts\\concepts.txt", "r", encoding='utf8')
lines = []
raw = []

#Preprocessing
for line in file.readlines():
	raw.append(line.replace("\n",""))
	for match in re.findall("[A-Z/_]", line):
		line = line.replace(match," " + match)
		line = line.replace("\n","")
		line = line.replace("_", "")
		line = line.replace("/", "")
		line = line.replace(" A ", " ")
		line = line.replace(" The ", " ")
		line = line.replace(" And ", " ")
		line = re.sub(' +', ' ', line)
	lines.append(line.strip().lower())
print("Loading all concepts ... [OK]")

def pos(tag):
 if tag.startswith('NN'):
  return wn.NOUN
 elif tag.startswith('V'):
  return wn.VERB

def synonyms(word, tag):
    lemma_lists = [ss.lemmas() for ss in wn.synsets(word, pos(tag))]
    lemmas = [lemma.name() for lemma in sum(lemma_lists, [])]
    return set(lemmas)

def computeSimilarity(queryPhrase, concept):
    score = 0
    queryWords = set(queryPhrase.split(" "))
    conceptWords = set(concept.split(" "))
    conceptSynonyms = set()
    for cw in conceptWords:
        conceptSynonyms.update(synonyms(cw, wn.NOUN))
        conceptSynonyms.update(synonyms(cw, wn.VERB))
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
    print(sentence)
    st = current_millis_time()
    doc = nlp(sentence)
    nouns = set()
    for nc in doc.noun_chunks:
        nouns.add(str(nc.text))
        nouns.add(lemmatizer.lemmatize(str(nc.text)))

    verbs = set()
    for possible_subject in doc:
        if possible_subject.head.pos == VERB:
            verbs.add(str(possible_subject.head))
            verbs.add(lemmatizer.lemmatize(str(possible_subject.head)))

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
    numConcepts = len(lines)

    #Similarity Matrix
    S = np.zeros((numConcepts,numPhrases))

    for i in range(0,numConcepts):
        for j in range(0,numPhrases):
            ss = computeSimilarity(phrases[j],lines[i])
            S[i][j] = ss
    sMax = S.max(0)
    for j in range(0, numPhrases):
        for i in range(0, numConcepts):
            if sMax[j] == S[i][j]:
                S[i][j] = sMax[j]
            else:
                S[i][j] = 0
    conceptVector = S.max(1)
    i=0
    listOfConcepts = []
    conceptScores = []
    for line in raw:
        if conceptVector[i] != 0:
            listOfConcepts.append(line)
            conceptScores.append(conceptVector[i])
        i = i + 1
    print("Computed in %d millisecs" % (current_millis_time() - st))
    print(str(listOfConcepts))
    print(str(conceptScores))