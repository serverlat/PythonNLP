import nltk 
from nltk import grammar, parse

dog = ["this", "dog", "runs"]
dogs = ["these", "dogs", "run"]
nonsense = ["there", "is", "cat"]

grammar1 = nltk.CFG.fromstring("""
    S -> NP V
    NP -> Det N
    Det -> "this" | "these" 
    N -> "dog" | "dogs" 
    V -> "run" | "runs" 
""")

def parse_sent(sentence):
    parser = nltk.RecursiveDescentParser(grammar1)
    try:
        trees = list(parser.parse(sentence))
        for tree in trees:
            print(tree)
        return trees[0]
    except:
        print("Invalid syntax :(")


parse_sent(nonsense)

grammar2 = """
    % start S
    S -> NP[NUM=?n] V[NUM=?n]
    NP[NUM=?n] -> Det[NUM=?n] N[NUM=?n]
    Det[NUM=sg] -> "this" 
    Det[NUM=pl] -> "these"
    N[NUM=sg] -> "dog"
    N[NUM=pl] -> "dogs" 
    V[NUM=sg] -> "runs" 
    V[NUM=pl] -> "run" 
"""
grammar3 = grammar.FeatureGrammar.fromstring(grammar2)
parser2 = parse.FeatureEarleyChartParser(grammar3)
trees = parser2.parse(dog)
for tree in trees: print(tree)