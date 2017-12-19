import time
import re
import numpy as np
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.stem.snowball import SnowballStemmer

# Data directory
data_dir = "C:\\Users\\myste\\Google Drive\\CMU\\Sem1\\Research\\Data\\AVS\\AVS_concept_detection\\AVS_concept_detection\\"
# Stop words, can be ignored from the query
cached_stop_words = stopwords.words("english")
# Lemmatizer
lemmatizer = WordNetLemmatizer()
stemmer = SnowballStemmer("english")
# To calculate time
current_milli_time = lambda: int(round(time.time() * 1000))


def stem(word):
    return stemmer.stem(word)


def getProcessedConcepts(path_to_concept_file = "C:\\Users\\myste\\Google Drive\\CMU\\Sem1\\Research\\Concepts\\concepts.txt"):
    file = open(path_to_concept_file, "r", encoding='utf8')
    concepts = []
    raw = []
    # Pre processing
    for line in file.readlines():
        raw.append(line.replace("\n", ""))
        for match in re.findall("[A-Z/_]", line):
            line = line.replace(match, " " + match)
            line = line.replace("\n", "")
            line = line.replace("_", "")
            line = line.replace("-", "")
            line = line.replace("(", "")
            line = line.replace(")", "")
            line = line.replace("/", "")
            line = re.sub(' +', ' ', line)
        wordsInConcept = [word for word in line.split() if word not in cached_stop_words]
        wordsInConcept = [lemmatizer.lemmatize(word) for word in wordsInConcept]
        #wordsInConcept = [stem(word) for word in wordsInConcept]
        concepts.append(create_query(wordsInConcept))
    return concepts, raw


def create_query(words):
    query = ""
    for word in words:
        query += word + " "
    return query.strip().lower()


def getQueries():
    query_file = "queries.txt"
    # Loading the queries
    file = open(data_dir+query_file, "r", encoding='utf8')
    queries = []
    for line in file.readlines():
        lines = line.split(":")
        query = lines[1].replace("Find shots of ", "")
        query = query.replace("\n", "")
        wordsInQuery = [word for word in query.split() if word not in cached_stop_words]
        wordsInQuery = [lemmatizer.lemmatize(word) for word in wordsInQuery]
        #wordsInQuery = [stem(word) for word in wordsInQuery]
        queries.append((lines[0], create_query(wordsInQuery)))
        #queries.append((lines[0], query))
    return queries


def getQueriesRaw():
    query_file = "queries.txt"
    # Loading the queries
    file = open(data_dir+query_file, "r", encoding='utf8')
    queries = []
    for line in file.readlines():
        lines = line.split(":")
        query = lines[1].replace("\n", "")
        queries.append((lines[0], query))
    return queries


def conceptScoresInVideos():
    # scores_sin = "scores_SIN_svm.txt"
    # file = open(data_dir + scores_sin, "r", encoding="utf8")
    # scores_sin = []
    # for line in file.readlines():
    #     scores_sin.append(np.array(list(map(float, line.split(" ")))))
    # scores_image_net = "scores_ImageNet.txt"
    # file = open(data_dir + scores_image_net, "r", encoding="utf8")
    # scores_image_net = []
    # for line in file.readlines():
    #     scores_image_net.append(np.array(list(map(float, line.split(" ")))))
    # scores = []
    # if len(scores_sin) != len(scores_image_net):
    #     print("Different number of video shots !!!")
    # for i in range(0, len(scores_sin)):
    #     scores.append(np.concatenate((scores_sin[i], scores_image_net[i][2:])))
    # np.save('scores_sin_image_net.npy', scores)
    scores = np.load('scores_sin_image_net.npy')
    return scores


def getSINconcepts():
    print("Extracting SIN concepts....")
    concepts_sin = "concept_names_SIN.txt"
    file = open(data_dir + concepts_sin, "r", encoding='utf8')
    concepts = []
    for line in file.readlines():
        line = line.split(":")[1].strip()
        line = line.replace("_", " ")
        line = line.replace("\n", "")
        wordsInConcept = [word for word in line.split() if word not in cached_stop_words]
        wordsInConcept = [lemmatizer.lemmatize(word) for word in wordsInConcept]
        concepts.append(create_query(wordsInConcept))
    print("Extracting SIN concepts....[OK]")
    return concepts


def getImagenetAndSINconcepts():
    concepts_sin = "concept_names_SIN.txt"
    file = open(data_dir + concepts_sin, "r", encoding='utf8')
    concepts = []
    for line in file.readlines():
        line = line.split(":")[1].strip()
        line = line.replace("_", " ")
        line = line.replace("\n", "")
        concepts.append(line)
    concepts_image_net = "concept_names_ImageNet.txt"
    file = open(data_dir + concepts_image_net, "r", encoding='utf8')
    for line in file.readlines():
        line = line.split(":")[1].strip()
        line = line.replace("_", " ")
        line = line.replace("\n", "")
        concepts.append(line)
    return concepts


def allConceptScoresInVideos():
    # scores = []
    # for filename in os.listdir(data_dir+"semantics_new"):
    #     score = np.zeros(348)
    #     file = str(filename).replace("shot","")
    #     file = file.replace(".json","")
    #     video_number = int(file.split("_")[0])
    #     shot_number = int(file.split("_")[1])
    #     score[0] = video_number
    #     score[1] = shot_number
    #     i = 2
    #     with open(data_dir+"semantics_new\\"+filename) as json_data:
    #         d = json.load(json_data)
    #         for elem in d['sports487']:
    #             score[i] = float(elem['score'])
    #             i += 1
    #         for elem in d['kinetics']:
    #             score[i] = float(elem['score'])
    #             i += 1
    #         for elem in d['sin346']:
    #             score[i] = float(elem['score'])
    #             i += 1
    #         for elem in d['places365']:
    #             score[i] = float(elem['score'])
    #             i += 1
    #         for elem in d['fcvid']:
    #             score[i] = float(elem['score'])
    #             i += 1
    #         for elem in d['ucf101']:
    #             score[i] = float(elem['score'])
    #             i += 1
    #         for elem in d['yfcc609']:
    #             score[i] = float(elem['score'])
    #             i += 1
    #         scores.append(score)
    # np.save('scoresSIN.npy', scores)
    # np.save('scoresNew.npy', scores)
    # np.save('scoresNewSIN.npy', scores)
    scores = np.load('scoresNew.npy')
    return scores
