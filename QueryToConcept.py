from ConceptVectors import getConcepts

query1 = "a girl cleaning utensils"
query2 = "a girl washing floor"
query3 = "some boy climbing rocks"
query4 = "baby sleeping in pajamas"
query5 = "baby crawling in pajamas"
query6 = "man hugging a women"
query7 = "Show me something blue"
query8 = "When was the last time Jack tried skydiving?"
query9 = "how many times has Jack travel to mountains?"
#things to try: dog, rifle, squash, squeeze,green grass everywhere

print("-------------------------------")
getConcepts(query1)
print("-------------------------------")
getConcepts(query2)
print("-------------------------------")
getConcepts(query3)
print("-------------------------------")
getConcepts(query4)
print("-------------------------------")
getConcepts(query5)
print("-------------------------------")
getConcepts(query6)
print("-------------------------------")
getConcepts(query7)
print("-------------------------------")
getConcepts(query8)
print("-------------------------------")
getConcepts(query9)
print("-------------------------------")
while True:
    query = input("Enter your query: ")
    getConcepts(query)
    print("-------------------------------")