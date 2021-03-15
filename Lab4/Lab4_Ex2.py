import nltk 
import re

def chunk(sentence):
    grammar = r"""
    NP: {<DT>? <JJ>* <NN>*} 
    NN: {<NN> <NN>}
    NNP: {<NNP>*}"""
    parser = nltk.RegexpParser(grammar)
    result = parser.parse(sentence)
    return result

text = []

with open("Lab4/SpaceX.txt", "r") as f:
    for line in f.readlines():
        if line != "\n": 
            temp = line.strip()
            text.append(re.findall(r"[\w']+", temp)) # filtering out words only, but it messes up [s]illiest :(

tagged = []

for line in text:
    tagged.append(nltk.pos_tag(line))
       
labels = ["NP", "NN", "NNP"] 

for i in range(5):
    tree = chunk(tagged[i])
    print(tree)
    for subtree in tree.subtrees():
        if subtree.label() in labels and subtree.height() > 1:
            words = [leaf[0] for leaf in subtree.leaves()]
            print("Matching text:"," ".join(words), "\n")

    
