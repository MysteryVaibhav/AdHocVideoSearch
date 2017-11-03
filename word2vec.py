import gensim
import time
import queue as Q
from PreProcess import getProcessedConcepts
from PhraseVector import PhraseVector

# Load Google's pre-trained Word2Vec model.
print("Loading model...")
path_to_trained_model = 'C:\\Users\\myste\\Downloads\\GoogleNews-vectors-negative300.bin'
model = gensim.models.KeyedVectors.load_word2vec_format(path_to_trained_model, binary=True)
print("Loading model... [OK]")

# To calculate time
current_milli_time = lambda: int(round(time.time() * 1000))

# Getting all the concepts
phraseVectors = []
concepts = getProcessedConcepts()

# Getting word embeddings for all the concepts
for concept in concepts:
    phraseVectors.append(PhraseVector(model, concept))

# top_k: Retrieves top k most similar concepts
top_k = 5;


def word2VecBestMatch(inputQuery):
    inputQuery = inputQuery.lower()
    print("Finding best match for " + inputQuery)
    st = current_milli_time()
    queryVector = PhraseVector(model, inputQuery)
    idx = 0;
    q = Q.PriorityQueue();
    for conceptVector in phraseVectors:
        score = queryVector.CosineSimilarity(conceptVector.vector)
        q.put((-1 * score, concepts[idx]))
        idx += 1
    idx = 0
    print("Best {} matches using word vectors: ".format(top_k))
    while (not q.empty()) and idx < 5:
        idx += 1;
        print(q.get());
    print("Found in {} milli secs".format(current_milli_time() - st))


# Testing
query1 = "hgfdgfkhuflkdflu"
query2 = "No white trains"
query3 = "Regular soccer match"
query4 = "chicken breasts"
query5 = "buildings and trees"
query6 = "bikes"
word2VecBestMatch(query1)
print('----------------------------')
word2VecBestMatch(query2)
print('----------------------------')
word2VecBestMatch(query3)
print('----------------------------')
word2VecBestMatch(query4)
print('----------------------------')
word2VecBestMatch(query5)
print('----------------------------')
word2VecBestMatch(query6)
while True:
    query = input("Enter your query: ")
    word2VecBestMatch(query)
    print("-------------------------------")
