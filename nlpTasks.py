import en_core_web_sm
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet as wn
from spacy.symbols import VERB

lemmatizer = WordNetLemmatizer()
print("Loading NLP ...")
nlp = en_core_web_sm.load()
print("Loading NLP ... [OK]")


def pos(tag):
    if tag.startswith('NN'):
        return wn.NOUN
    elif tag.startswith('V'):
        return wn.VERB


def getSynonyms(word, tag):
    lemma_lists = [ss.lemmas() for ss in wn.synsets(word, pos(tag))]
    lemmas = [lemma.name() for lemma in sum(lemma_lists, [])]
    return set(lemmas)


def getVerbs(sentence):
    doc = nlp(sentence)
    verbs = set()
    for possible_subject in doc:
        if possible_subject.head.pos == VERB:
            verbs.add(str(possible_subject.head))
            verbs.add(lemmatizer.lemmatize(str(possible_subject.head)))
    return verbs


def getNouns(sentence):
    doc = nlp(sentence)
    nouns = set()
    for nc in doc.noun_chunks:
        nouns.add(str(nc.text))
        nouns.add(lemmatizer.lemmatize(str(nc.text)))
    return nouns
