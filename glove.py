import gensim
import time
import queue as Q
from PreProcess import getProcessedConcepts
from PhraseVector import PhraseVector

# Load Stanford's pre-trained glove model.
print("Loading model...")
path_to_trained_model = 'C:\\Users\\myste\\Downloads\\glove.6B\\glove.6B.100d.txt.word2vec'
model = gensim.models.KeyedVectors.load_word2vec_format(path_to_trained_model, binary=False)
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


def gloveBestMatch(inputQuery):
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
gloveBestMatch(query1)
print('----------------------------')
gloveBestMatch(query2)
print('----------------------------')
gloveBestMatch(query3)
print('----------------------------')
gloveBestMatch(query4)
print('----------------------------')
gloveBestMatch(query5)
print('----------------------------')
gloveBestMatch(query6)
while True:
    query = input("Enter your query: ")
    gloveBestMatch(query)
    print("-------------------------------")
