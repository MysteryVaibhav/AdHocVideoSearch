import numpy as np
import subprocess
import queue as Q
from nlpTasks import getConstituents, getNounsAndVerbs
from PreProcess import current_milli_time, getProcessedConcepts, data_dir, lemmatizer, cached_stop_words, create_query, getImagenetAndSINconcepts
from tfIdf import idf1
from sematic_role_labelling import getRelevantSubQueries

concepts = getImagenetAndSINconcepts()#getProcessedConcepts(path_to_concept_file=data_dir + "AllConcepts.txt")
threshold = 0.8
process_esa = "java -Xmx1024m -cp lib\*;esalib.jar clldsystem.esa.ESAAnalyzer"
p_esa = subprocess.Popen(process_esa, stdin=subprocess.PIPE,
                            stdout=subprocess.PIPE, universal_newlines=True, encoding="utf8")
file = open(data_dir + "esaBestMatches_sin_imagenet.txt", "w", encoding="utf8")
#filePair = open(data_dir + "wordPairs_sin_imagenet.txt", "w", encoding="utf8")
filePair = open(data_dir + "wordPairs_sin_imagenet.txt", "r", encoding="utf8")
esaScores = {}
for row in filePair.readlines():
    elems = row.split("*")
    if elems[0] not in esaScores:
        esaScores[elems[0]] = float(elems[1])


def esa_similarity(term1, term2):
    if term1 + ":" + term2 in esaScores:
        return esaScores[term1 + ":" + term2]

    p_esa.stdin.write(term1+"\n")
    p_esa.stdin.write(term2+"\n")
    p_esa.stdin.flush()
    score = 0
    while True:
        try:
            score = float(p_esa.stdout.readline())
            if score != 3:  #Because of the concept '3 or more people'
                break
        except ValueError:
            pass
    return score


# Measure ESA similarity
def step1(query):
    concept_vector = np.zeros(len(concepts))
    matched = False
    for i in range(0,len(concepts)):
        score = esa_similarity(query, concepts[i])
        #filePair.write(query + ":" + concepts[i] + "*" + str(score) +"\n")
        if score > threshold:
            concept_vector[i] = score
            matched = True
    return matched, concept_vector


# Exact Match
def step2(query):
    matched_concepts_scores = {}
    for concept in concepts:
        score = 0
        concept_set = set(concept.split(" "))
        for word in query.split(" "):
            if word in cached_stop_words or word == "Find" or word == "shots":
                continue
            if concept_set.__contains__(word):
                score += idf1(word)
        if score > 0:
            matched_concepts_scores[concept] = score
    query += " "
    for concept in concepts:
        if concept + " " in query:
            matched_concepts_scores[concept] = idf1(concept)
    return matched_concepts_scores


# Sub query matching
def step3(sub_queries, matched_concept_scores):
    for sub_query in sub_queries:
        q = Q.PriorityQueue()
        for concept in concepts:
            score = esa_similarity(sub_query, concept)
            #filePair.write(sub_query + ":" + concept + "*" + str(score) + "\n")
            if score == score:
                q.put((-1*score, concept))
        while not q.empty():
            elem = q.get()
            if -1*elem[0] > threshold and elem[1] not in matched_concept_scores:
                matched_concept_scores[elem[1]] = -1*elem[0]#*idf1(elem[1])
            if -1*elem[0] < threshold:
                break
    return matched_concept_scores


def get_matched_concepts_from_concept_vector(concept_vector):
    matched_concepts = []
    concept_scores = []
    for i in range(0, len(concept_vector)):
        if concept_vector[i] > 0:
            matched_concepts.append(concepts[i])
            concept_scores.append(concept_vector[i])
    return matched_concepts, concept_scores


def match(query):
    st = current_milli_time()
    query = query.replace(" any type of ", " ").lower()
    # Step1
    matched, concept_vector_step1 = step1(query)
    if matched:
        matched_concepts, concept_scores = get_matched_concepts_from_concept_vector(concept_vector_step1)
        print(query + " :: " + str(matched_concepts))
        file.write("Query ::" + query + "\n")
        file.write("Matches ::" + str(matched_concepts) + "\n")
        file.write("Scores ::" + str(concept_scores) + "\n")
        file.write("--------------------------------------------\n")
        print(concept_scores)
        print("Computed in {} milli secs".format(current_milli_time() - st))
        return concept_vector_step1

    # Step2
    matched_concept_scores = {}
    matched_concept_scores = step2(query)

    # Step3
    sub_queries = getRelevantSubQueries(query)
    matched_concept_scores = step3(sub_queries, matched_concept_scores)
    concept_vector = np.zeros(len(concepts))
    matched_concepts = []
    concept_scores = []
    for i in range(0, len(concept_vector)):
        if concepts[i] in matched_concept_scores:
            concept_vector[i] = matched_concept_scores[concepts[i]]
            matched_concepts.append(concepts[i])
            concept_scores.append(matched_concept_scores[concepts[i]])
    file.write("Query ::" + query + "\n")
    file.write("Matches ::" + str(matched_concepts) + "\n")
    file.write("Scores ::" + str(concept_scores) + "\n")
    file.write("--------------------------------------------\n")
    return concept_vector
