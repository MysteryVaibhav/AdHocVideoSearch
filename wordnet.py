import re
from nltk.corpus import wordnet as wn
import time

current_milli_time = lambda: int(round(time.time() * 1000))

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

#Filling up synsets for each concept
synsets = []
for line in lines:
    sublist = []
    for word in line.split(" "):
        for synset in wn.synsets(word):
            sublist.append(synset)
            break;
    synsets.append(sublist)
print(synsets)

def wupBestMatch(query):
    query = query.lower()
    print("Finding best match for " + query)
    st = current_milli_time()
    querySyns = []
    for queryWord in query.split(" "):
        for querySyn in wn.synsets(queryWord):
            querySyns.append(querySyn)
            break;

    # Finding similar concept
    maxScoreWup = 0
    maxIdxWup = 0
    idx = 0
    for synList in synsets:
        scoreWup = 0
        noMatchWup = 0
        for querySyn in querySyns:
            for syn in synList:
                indScore = wn.wup_similarity(querySyn, syn);
                if (indScore != None):
                    scoreWup += indScore
                else:
                    noMatchWup += 1
        if len(querySyns) * len(synList) - noMatchWup > 0:
            scoreWup /= len(querySyns) * len(synList) - noMatchWup
            if maxScoreWup < scoreWup:
                maxScoreWup = scoreWup
                maxIdxWup = idx
        idx += 1
    print("Best match Wup similarity: " + lines[maxIdxWup] + " :: ", maxScoreWup)
    print("Found in %d millisecs" % (current_milli_time() - st))

def lchBestMatch(query):
    query = query.lower()
    print("Finding best match for " + query)
    st = current_milli_time()
    querySyns = []
    for queryWord in query.split(" "):
        for querySyn in wn.synsets(queryWord):
            querySyns.append(querySyn)
            break;

    #Finding similar concept
    maxScoreLch = 0
    maxIdxLch = 0
    idx = 0
    for synList in synsets:
        scoreLch = 0
        noMatchLch = 0
        for querySyn in querySyns:
            for syn in synList:
                indScore = 0
                if (querySyn._pos == syn._pos):
                    indScore = wn.lch_similarity(querySyn, syn)
                if (indScore != None):
                    scoreLch += indScore
                else:
                    noMatchLch += 1
        if len(querySyns)*len(synList) - noMatchLch > 0:
            scoreLch /= len(querySyns)*len(synList) - noMatchLch
            if maxScoreLch < scoreLch:
                maxScoreLch = scoreLch
                maxIdxLch = idx
        idx += 1
    print("Best match Lch similarity: " + lines[maxIdxLch] + " :: ",maxScoreLch)
    print("Found in %d millisecs" % (current_milli_time() - st))

##Testing

query1 = "hgfdgfkhuflkdflu"
query2 = "No white trains"
query3 = "Regular soccer match"
query4 = "chicken breasts"
query5 = "buildings and trees"
query6 = "bikes"
wupBestMatch(query1)
lchBestMatch(query1)
print('----------------------------')
wupBestMatch(query2)
lchBestMatch(query2)
print('----------------------------')
wupBestMatch(query3)
lchBestMatch(query3)
print('----------------------------')
wupBestMatch(query4)
lchBestMatch(query4)
print('----------------------------')
wupBestMatch(query5)
lchBestMatch(query5)
print('----------------------------')
wupBestMatch(query6)
lchBestMatch(query6)