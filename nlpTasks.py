import gensim.models.word2vec as w2v
from stanfordcorenlp import StanfordCoreNLP
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet as wn
from spacy.symbols import VERB
from PreProcess import cached_stop_words, create_query

lemmatizer = WordNetLemmatizer()
print("Loading NLP ...")
nlp = StanfordCoreNLP(r"C:\\Users\\myste\\Downloads\\stanford-corenlp-full-2017-06-09\\stanford-corenlp-full-2017-06-09")
print("Loading NLP ... [OK]")
print("Loading model...")
path_to_trained_model = 'myModel'
model = w2v.Word2Vec.load(path_to_trained_model)
print("Loading model... [OK]")


def pos(tag):
    if tag.startswith('NN'):
        return wn.NOUN
    elif tag.startswith('V'):
        return wn.VERB


def getSynonyms(word, tag):
    lemma_lists = [ss.lemmas() for ss in wn.synsets(word, pos(tag))]
    lemmas = [lemma.name() for lemma in sum(lemma_lists, [])]
    return set(lemmas)


def getDependencyParse(sentence):
    return nlp.dependency_parse(sentence)


def getSimilarWords(word):
    similar_words = set()
    if word in model.wv.vocab:
        for pair in model.wv.most_similar(word, topn=3):
            if pair[1] > 0.6:
                similar_words.add(pair[0])
            else:
                break
    return similar_words


def getNounsAndVerbs(sentence):
    nouns = set()
    verbs = set()
    for pair in nlp.pos_tag(sentence):
        if pair[1].startswith('N'):
            nouns.add(pair[0])
        if pair[1].startswith('V'):
            verbs.add(pair[0])
    return nouns, verbs


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


#TODO
def getConstituents(sentence):
    wordsInQuery = [word for word in sentence.split() if word not in cached_stop_words]
    wordsInQuery = [lemmatizer.lemmatize(word) for word in wordsInQuery]
    nouns, verbs = getNounsAndVerbs(create_query(wordsInQuery))
    sub_queries = []
    for verb in verbs:
        for noun in nouns:
            sub_queries.append(verb + " " + noun)
    sub_queries.extend(nouns)
    sub_queries.extend(verbs)
    return sub_queries


def getPosTags(sentence):
    return nlp.pos_tag(sentence)
