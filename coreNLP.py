import json
from stanfordcorenlp import StanfordCoreNLP

#with open("C:\\Users\\myste\\Google Drive\\CMU\\Sem1\\Research\\Data\\expertvagabond_blog_posts.json", "r", encoding="utf-8") as f:
#    dataStore = json.load(f)

#for data in dataStore:
#    print(data["title"] + str(data["text"]))

print("Loading core NLP ...")
corenlp = StanfordCoreNLP(r"C:\\Users\\myste\\Downloads\\stanford-corenlp-full-2017-06-09\\stanford-corenlp-full-2017-06-09")
print("Loading core NLP ... [OK]")

print("\nCoreNlp Parse")
print(corenlp.parse("Fruit flies like the banana"))
print(corenlp.pos_tag("When was the last time Jack tried skydiving?"))
print(corenlp.pos_tag("When was the last time Jack ate apple cake"))
#print(corenlp.ner("When was the last time Jack ate apple cake"))
print(corenlp.dependency_parse("When was the last time Jack ate apple cake"))
print(corenlp.dependency_parse("I have a meeting with Mary at 1pm"))

nouns = set()
verbs = set()
for pair in corenlp.pos_tag("When was the last time Jack tried skydiving?"):
    if pair[1].startswith('NN'):
        nouns.add(pair[0])
    if pair[1].startswith('V'):
        verbs.add(pair[0])

print(nouns)
print(verbs)