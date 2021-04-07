from collections import Counter
from spacy import displacy
import spacy
import pickle
import matplotlib.pyplot as plt
import nltk
from nltk.probability import FreqDist

nlp = spacy.load("en_core_web_sm")

book = []

with open("Lab5/fellowship.txt", "r", encoding="utf8") as rawBook:
    book = rawBook.read()

# First time run:
#book = book.replace("\n\n", "")
#doc = nlp(book)
#entities = [(e.text, e.label_, e.kb_id_) for e in doc.ents]
#pickle.dump(entities, open( "entities.p", "wb" ) )

# Pickle for convenience
entities = pickle.load(open( "entities.p", "rb"))
counter = Counter([entry[0] for entry in list(filter((lambda x: (x[1] == 'PERSON')), entities))]) 

top = {}
for entry in counter.most_common(25):
    top[entry[0]] = entry[1]
plt.bar(top.keys(), top.values())
plt.xticks(rotation=90)
plt.show()
