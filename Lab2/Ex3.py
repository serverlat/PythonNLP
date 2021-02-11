import nltk 
from nltk.tag import UnigramTagger
from nltk.corpus import brown

#a

fdist = nltk.FreqDist(brown.words())
cfdist = nltk.ConditionalFreqDist(brown.tagged_words())
top_words = fdist.most_common(200)
most_likely_tags = dict((word, cfdist[word].max()) for (word, _) in top_words) #from nltk ch 5
default_tagger = UnigramTagger(model=most_likely_tags)

splits = [[90,10], [50,50]]
correct_brown = brown.tagged_sents()

for split in splits: 
    size_brown = int(len(correct_brown)*(split[0]/100))
    train_brown = correct_brown[:size_brown] #up to 90%
    test_brown = correct_brown[size_brown:] #from 90% to 100%

    print(f"The LookupTagger accuracy for the Brown Corpus is {default_tagger.evaluate(test_brown)} using a {split[0]}/{split[1]} split.")
    #50% correct as expected 
  