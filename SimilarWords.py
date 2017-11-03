import gensim
import warnings
import re
warnings.filterwarnings(action='ignore', category=UserWarning, module='gensim')

import time

import numpy as np
import math
from nltk.corpus import stopwords

# Load Google's pre-trained Word2Vec model.
print ("Loading the model...")
model = gensim.models.KeyedVectors.load_word2vec_format('C:\\Users\\myste\\Downloads\\GoogleNews-vectors-negative300.bin', binary=True)
print ("Loading the model... [OK]")

file = open("C:\\Users\\myste\\Google Drive\\CMU\\Sem1\\Research\\Concepts\\concepts.txt", "r", encoding='utf8')
lines = []

#Preprocessing
for line in file.readlines():
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
print(lines)

f = open("C:\\Users\\myste\\Google Drive\\CMU\\Sem1\\Research\\Concepts\\similarWords.txt", "w", encoding='utf8')
for concept in lines:
    if model.__contains__(concept):
        f.write(concept + " ::nearest neigbors:: ")
        f.write(str(model.most_similar(concept)))
        f.write("\n")
    else:
        f.write(concept + " ::nearest neigbors:: phrase not found in model\n")
f.close()
