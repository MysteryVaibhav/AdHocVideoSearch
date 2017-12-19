from PreProcess import getQueriesRaw, cached_stop_words
from nlpTasks import lemmatizer, nlp


class semantic_frame:

    def __init__(self, query):

        # Initializing variables
        self.text = query.replace(",", " ,")
        self.words = self.text.split(" ")
        self.parse = nlp.dependency_parse(self.text)#getDependencyParse(self.text)
        self.posTags = nlp.pos_tag(self.text)
        self.agent = []
        self.object = []
        self.activity = []
        self.space = []
        self.time = []
        self.locations = set(['indoor', 'outdoor', 'city', 'street', 'stage', 'bridge'])
        self.times = set(['day', 'night', 'time'])

        # Computing
        self.fill_roles()
        self.post_processing()

    def __str__(self):
        pprint = '-------------------------------------------\n'
        pprint += "Query: " + self.text + "\n"
        pprint += "Parse: " + str(self.parse) + "\n"
        pprint += "Pos-tags: " + str(self.posTags) + "\n"
        pprint += "Activity: " + str(self.activity) + "\n"
        pprint += "Agent: " + str(self.agent) + "\n"
        pprint += "Object: " + str(self.object) + "\n"
        pprint += "Space: " + str(self.space) + "\n"
        pprint += "Time: " + str(self.time) + "\n"
        pprint += '-------------------------------------------'
        return pprint

    def get_compound_chain(self, idx):
        word = ""
        for location in self.locations:
            if location in self.words[idx]:
                self.space.append(self.words[idx])
                idx -= 1
                break
        while idx > 0 and self.posTags[idx-1][1].startswith('N'):
            word = self.words[idx] + " " + word
            idx -= 1
        return word, idx

    def get_adjective_chain(self, word, idx):
        while idx > 0 and self.posTags[idx-1][1].startswith('J'):
            word = self.words[idx] + " " + word
            idx -= 1
        return word

    def set_agent(self, idx):
        word, new_idx = self.get_compound_chain(idx)
        if idx - new_idx == 1:
            word = self.get_adjective_chain(word, new_idx)
        self.agent.append(word)

    def set_activity(self, idx):
        self.activity.append(self.words[idx])

    def set_object(self, idx):
        word, new_idx = self.get_compound_chain(idx)
        if idx - new_idx == 1:
            word = self.get_adjective_chain(word, new_idx)
        self.object.append(word)

    def set_time(self, idx):
        for time in self.times:
            if time in self.words[idx]:
                self.time.append(self.words[idx])
                return True
        return False

    def set_space(self, idx):
        for location in self.locations:
            if location in self.words[idx]:
                self.space.append(self.words[idx])
                return True
        return False

    def fill_roles(self):
        for i in range(2, len(self.parse)):
            node = self.parse[i]
            if node[0] == 'ROOT':
                self.set_activity(node[2])
            if node[0] == 'nsubj':
                self.set_agent(node[2])
            if node[0] == 'dobj':
                self.set_object(node[2])
            if node[0] == 'acl':
                self.set_activity(node[2])
            if node[0] == 'xcomp':
                self.set_activity(node[2])
            if node[0] == 'amod' and self.posTags[node[2]-1][1].startswith('V'):
                self.set_activity(node[2])
            if node[0] == 'advmod' or node[0] == 'advcl':
                self.set_time(node[2])
                self.set_space(node[2])
            if node[0] == 'nmod':
                if not self.set_time(node[2]) and not self.set_space(node[2]):
                    self.set_agent(node[2])
            if node[0] == 'dep':
                if self.posTags[node[2]-1][1].startswith('N'):
                    self.set_agent(node[2])
                if self.posTags[node[2]-1][1].startswith('V'):
                    self.set_activity(node[2])
            if node[0] == 'conj':
                if self.posTags[node[2]-1][1].startswith('N') and not self.words[node[2]].endswith('ing'):
                    self.set_agent(node[2])
                if self.posTags[node[2]-1][1].startswith('V') or (self.posTags[node[2]-1][1].startswith('N') and self.words[node[2]].endswith('ing')):
                    self.set_activity(node[2])

    def post_processing(self):
        self.activity = [word.strip().lower() for word in self.activity]
        self.agent = [word.strip().lower() for word in self.agent]
        self.object = [word.strip().lower() for word in self.object]
        self.space = [word.strip().lower() for word in self.space]
        self.time = [word.strip().lower() for word in self.time]

        self.activity = [lemmatizer.lemmatize(word) for word in self.activity if word not in cached_stop_words and word != '']
        self.agent = [lemmatizer.lemmatize(word) for word in self.agent if word not in cached_stop_words and word != '']
        self.object = [lemmatizer.lemmatize(word) for word in self.object if word not in cached_stop_words and word != '']
        self.space = [lemmatizer.lemmatize(word) for word in self.space if word not in cached_stop_words and word != '']
        self.time = [lemmatizer.lemmatize(word) for word in self.time if word not in cached_stop_words and word != '']


def getRelevantSubQueries(sentence):
    sf = semantic_frame(sentence)
    return set(sf.activity+sf.agent+sf.space+sf.time+sf.object)


if __name__ == '__main__':
    queries = getQueriesRaw()
    for query in queries:
        sf = semantic_frame(query[1])
        print(sf)
