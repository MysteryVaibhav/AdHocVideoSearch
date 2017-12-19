import json
import gensim
import gensim.models.word2vec as w2v
from PreProcess import data_dir, cached_stop_words, lemmatizer

with open(data_dir + "captions_train2014.json", "r", encoding="utf-8") as f:
    dataStore = json.load(f)

sentences = []
for annotation in dataStore['annotations']:
    sentence = annotation['caption'].lower()
    sentence = sentence.replace("\n", "")
    sentence = sentence.replace(".", "")
    sentence = sentence.replace("!", "")
    sentence = sentence.replace("\"", "")
    wordsInSentence = [word for word in sentence.split() if word not in cached_stop_words]
    wordsInSentence = [lemmatizer.lemmatize(word) for word in wordsInSentence]
    sentences.append(wordsInSentence)

print("Number of sentences: " + str(len(sentences)))
model = w2v.Word2Vec(sentences=sentences, size=50, alpha=0.025, window=3, min_count=1,
                     workers=4, min_alpha=0.0001, iter=5)


model.save("MyModel1")

print(model['birthday'])
print(len(model.wv.vocab))
print(model.wv.most_similar('birthday'))
print(model.wv.most_similar('candle'))
print(model.wv.most_similar('golf'))
print(model.wv.most_similar('person'))