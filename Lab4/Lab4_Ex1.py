import nltk 
from nltk.corpus import brown

tagged_brown = brown.tagged_sents()

def chunk(sentence):
    grammar = r"""
    NP: {<DT>? <JJ>* <NN>*} # NP
    P: {<IN>} # Preposition
    V: {<V.*>} # Verb
    PP: {<P> <NP>} # PP -> P NP
    VP: {<V> <NP|PP>*} # VP -> V (NP|PP)*# Noun phrase
    CLAUSE:{<VB> <NP>} # Verb"""
    parser = nltk.RegexpParser(grammar)
    result = parser.parse(sentence)
    return result

tuples = set()

for sent in tagged_brown:
    tree = chunk(sent)
    for subtree in tree.subtrees():
        if subtree.label() == "VP" and len(subtree.leaves()) > 1:
            words = []
            for leaf in subtree.leaves():
                words.append(leaf[0])
            tuples.add(tuple(words))
    if len(tuples) == 20:
        break

print("20 results: ")
for sentence in tuples:
    print(" ".join(sentence))