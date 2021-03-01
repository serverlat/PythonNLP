import nltk
from nltk import FreqDist, NaiveBayesClassifier
from nltk.corpus import movie_reviews
import random
from nltk.corpus import wordnet as wn

documents = [(list(movie_reviews.words(fileid)), category)
             for category in movie_reviews.categories()
             for fileid in movie_reviews.fileids(category)]
all_words = FreqDist(w.lower() for w in movie_reviews.words())
word_features = list(all_words)[:2000]


def document_features(document):
    document_words = set(document)
    features = {} # use wordnet to check if review contains synonyms to the word?
    #dummycount = 0
    for word in word_features:
        if len(word) > 1 and word.isalpha():
            synsets= wn.synsets(word)
            if synsets:
                word = synsets[0].hypernyms()[0].lemmas()[0].name() if synsets[0].hypernyms() else word
            features['contains({})'.format(word)] = (word in document_words)
            #dummycount +=1
            #if dummycount > 20: break
        #features['contains({})'.format(word)] = (word in document_words)
    return features


#print(document_features(movie_reviews.words('pos/cv957_8737.txt')))

featuresets = [(document_features(d), c) for (d, c) in documents]
train_set, test_set = featuresets[100:], featuresets[:100]
classifier = NaiveBayesClassifier.train(train_set)

print(nltk.classify.accuracy(classifier, test_set))
classifier.show_most_informative_features(5)
