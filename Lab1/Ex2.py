import nltk
nltk.download("all")
from nltk.corpus import state_union

# Exercise 2

# a
files = list(state_union.fileids())
terms = ["men", "women", "people"]
statistics = nltk.ConditionalFreqDist((file, word)
                                      for file in state_union.fileids()
                                      for word in state_union.words(file)
                                      for term in terms if word.lower() == term)
statistics.tabulate(conditions=files, samples=terms)

# b
years_raw = sorted(list(set([int(year[:4]) for year in state_union.fileids()])))
years = [str(year) for year in years_raw]
year_statistics = nltk.ConditionalFreqDist((word.lower(), fileid[:4])
                                           for fileid in state_union.fileids()
                                           for word in state_union.words(fileid)
                                           for term in terms
                                           if word.lower() == term)
year_statistics.plot()
# More women over time, a lot of people in 1995 and 1946, more or less stable amount of men.
