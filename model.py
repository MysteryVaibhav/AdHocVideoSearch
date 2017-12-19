import gensim
from stanfordcorenlp import StanfordCoreNLP
import time
import numpy as np
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet as wn
from nltk.corpus import stopwords, reuters
from nltk import word_tokenize
from properties import path_to_trained_model, path_to_concepts
from PhraseVector import PhraseVector


# Helper methods
def pos(tag):
    if tag.startswith('NN'):
        return wn.NOUN
    elif tag.startswith('V'):
        return wn.VERB


def getSynonyms(word, tag):
    key = word + ":" + str(pos(tag))
    if key in map_word_synonyms:
        return map_word_synonyms[key]
    lemma_lists = [ss.lemmas() for ss in wn.synsets(word, pos(tag))]
    lemmas = [lemma.name() for lemma in sum(lemma_lists, [])]
    return set(lemmas)


def getNounsAndVerbs(sentence):
    nouns = set()
    verbs = set()
    for pair in nlp.pos_tag(sentence):
        if pair[1].startswith('NN'):
            nouns.add(pair[0])
        if pair[1].startswith('V'):
            verbs.add(pair[0])
    print(nouns)
    print(verbs)
    return nouns, verbs


def getProcessedConcepts(path_to_concept_file):
    file = open(path_to_concept_file, "r", encoding='utf8')
    concepts = []
    concept_ids = []
    # Pre processing
    for line in file.readlines():
        concept_id, concept = line.split(" ")
        concept_ids.append(concept_id)
        concept = concept.replace("\n", "")
        concept = concept.replace("_", "")
        concept = concept.replace("-", "")
        concept = concept.replace("(", "")
        concept = concept.replace(")", "")
        concept = concept.replace("/", "")
        wordsInConcept = [word for word in concept.split() if word not in cached_stop_words]
        wordsInConcept = [lemmatizer.lemmatize(word) for word in wordsInConcept]
        concepts.append(createConcept(wordsInConcept))
    return concepts, concept_ids


def createConcept(words):
    concept = ""
    for word in words:
        concept += word + " "
    return concept.strip().lower()


def tokenize(text):
    words = word_tokenize(text)
    words = [w.lower() for w in words]
    return [w for w in words if w not in cached_stop_words and not w.isdigit()]


def tfidf(word):
    score = 0.1
    if word in word_index:
        score = word_idf[word_index[word]]
    return score


def stripCommonPhrases(sentence):
    sentence = sentence.replace("find videos of", "")
    sentence = sentence.replace("show videos of", "")
    sentence = sentence.replace("show me videos of", "")
    sentence = sentence.replace("find shots of", "")
    sentence = sentence.replace("show shots of", "")
    sentence = sentence.replace("people", "")
    sentence = sentence.replace("person", "")
    return sentence

# Load Stanford's pre-trained glove model
print("Loading word embeddings...")
model = gensim.models.KeyedVectors.load_word2vec_format(path_to_trained_model, binary=False)
print("Loading word embeddings... [OK]")

# To calculate time
current_milli_time = lambda: int(round(time.time() * 1000))

print("Loading Spacy module ...")
#nlp = en_core_web_sm.load()
nlp = StanfordCoreNLP(r"C:\\Users\\myste\\Downloads\\stanford-corenlp-full-2017-06-09\\stanford-corenlp-full-2017-06-09")
print("Loading Spacy module ... [OK]")
lemmatizer = WordNetLemmatizer()
cached_stop_words = stopwords.words("english")
semantic_concepts, concept_ids = getProcessedConcepts(path_to_concepts)

print("Building vocabulary to compute tf_idf score...")
# build the vocabulary in one pass
vocabulary = set()
for file_id in reuters.fileids():
    words = tokenize(reuters.raw(file_id))
    vocabulary.update(words)

vocabulary = list(vocabulary)
word_index = {w: idx for idx, w in enumerate(vocabulary)}

VOCABULARY_SIZE = len(vocabulary)
DOCUMENTS_COUNT = len(reuters.fileids())

word_idf = np.zeros(VOCABULARY_SIZE)
for file_id in reuters.fileids():
    words = set(tokenize(reuters.raw(file_id)))
    indexes = [word_index[word] for word in words]
    word_idf[indexes] += 1.0

word_idf = np.log(DOCUMENTS_COUNT / (1 + word_idf).astype(float))
print("Building vocabulary to compute tf_idf score... [OK]")

phraseVectors = []
for concept in semantic_concepts:
    phraseVectors.append(PhraseVector(model, concept))

map_word_synsets = {}
map_word_synonyms = {}
for word in model.vocab:
    map_word_synsets[word] = wn.synsets(word)
    map_word_synonyms[word + ":" + str(pos(wn.NOUN))] = getSynonyms(word, wn.NOUN)
    map_word_synonyms[word + ":" + str(pos(wn.VERB))] = getSynonyms(word, wn.VERB)
