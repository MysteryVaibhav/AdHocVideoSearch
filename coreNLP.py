import json
from stanfordcorenlp import StanfordCoreNLP

with open("C:\\Users\\myste\\Google Drive\\CMU\\Sem1\\Research\\Data\\expertvagabond_blog_posts.json", "r", encoding="utf-8") as f:
    dataStore = json.load(f)

for data in dataStore:
    print(data["title"] + str(data["text"]))

print("Loading core NLP ...")
corenlp = StanfordCoreNLP(r"C:\\Users\\myste\\Downloads\\stanford-corenlp-full-2017-06-09\\stanford-corenlp-full-2017-06-09")
print("Loading core NLP ... [OK]")

print("\nCoreNlp Parse")
print(corenlp.parse("Fruit flies like the banana"))