import gensim.models.word2vec as w2v

model = w2v.Word2Vec.load("MyModel1")
print(len(model.wv.vocab))
print(model.wv.most_similar('birthday'))
print(model.wv.most_similar('candle'))
print(model.wv.most_similar('golf'))
print(model.wv.most_similar('person'))