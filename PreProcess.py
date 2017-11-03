import re

path_to_concept_file = "C:\\Users\\myste\\Google Drive\\CMU\\Sem1\\Research\\Concepts\\concepts.txt";
file = open(path_to_concept_file, "r", encoding='utf8')


def getProcessedConcepts():
    concepts = []
    # Pre processing
    for line in file.readlines():
        for match in re.findall("[A-Z/_]", line):
            line = line.replace(match, " " + match)
            line = line.replace("\n", "")
            line = line.replace("_", "")
            line = line.replace("/", "")
            line = line.replace(" A ", " ")
            line = line.replace(" The ", " ")
            line = line.replace(" And ", " ")
            line = re.sub(' +', ' ', line)
        concepts.append(line.strip().lower())
    return concepts
