from hybridMatch import bestMatch

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
bestMatch(query1)
print("-------------------------------")
bestMatch(query2)
print("-------------------------------")
bestMatch(query3)
print("-------------------------------")
bestMatch(query4)
print("-------------------------------")
bestMatch(query5)
print("-------------------------------")
bestMatch(query6)
print("-------------------------------")
bestMatch(query7)
print("-------------------------------")
bestMatch(query8)
print("-------------------------------")
bestMatch(query9)
print("-------------------------------")
while True:
    query = input("Enter your query: ")
    bestMatch(query)
    print("-------------------------------")