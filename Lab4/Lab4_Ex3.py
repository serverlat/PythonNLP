import nltk 
from nltk import grammar, parse

dog = ["this", "dog", "runs"]
dogs = ["these", "dogs", "run"]
nonsense = ["there", "is", "cat"]

grammar = nltk.CFG.fromstring("""
    S -> NP V
    NP -> Det N
    Det -> "this" | "these" 
    N -> "dog" | "dogs" 
    V -> "run" | "runs" 
""")

def parse(sentence):
    parser = nltk.RecursiveDescentParser(grammar)
    try:
        trees = list(parser.parse(sentence))
        for tree in trees:
            print(tree)
        return trees[0]
    except:
        print("Invalid syntax :(")


parse(nonsense)
parse(dog)
parse(dogs)
