import nltk 
from nltk import grammar, parse

dog = ["this", "dog", "runs"]
dogs = ["these", "dogs", "run"]
invalid = ["these", "dog", "run"]
nonsense = ["there", "is", "cat"]

grammar = nltk.CFG.fromstring("""
    S -> NP_SG VP_SG
    S -> NP_PL VP_PL
    NP_SG -> Det_SG N_SG
    NP_PL -> Det_PL N_PL
    VP_SG -> V_SG
    VP_PL -> V_PL

    Det_SG -> 'this'
    Det_PL -> 'these'
    N_SG -> 'dog'
    N_PL -> 'dogs'
    V_SG -> 'runs'
    V_PL -> 'run' 
""")

def parse_sent(sentence):
    parser = nltk.RecursiveDescentParser(grammar)
    try:
        trees = list(parser.parse(sentence))
        for tree in trees:
            print(tree)
        return trees[0]
    except:
        print(f"The sentence '{' '.join(sentence)}' has invalid syntax :(")


parse_sent(nonsense)
parse_sent(dog)
parse_sent(dogs)
parse_sent(invalid)
