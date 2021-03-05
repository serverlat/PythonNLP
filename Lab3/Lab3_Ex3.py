import nltk
from nltk import FreqDist, NaiveBayesClassifier
from nltk.corpus import movie_reviews
import random
from nltk.corpus import wordnet as wn
from nltk.corpus import stopwords 
from nltk.stem import WordNetLemmatizer 

documents = [(list(movie_reviews.words(fileid)), category)
             for category in movie_reviews.categories()
             for fileid in movie_reviews.fileids(category)]
all_words = FreqDist(w.lower() for w in movie_reviews.words())
word_features = list(all_words)[:2000]
stop_words = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer() 

def document_features(document):
    document_words = set(document)
    features = {} 
    for word in word_features:
        features['contains({})'.format(word)] = (word in document_words)
        if len(word) > 2:
            synsets = wn.synsets(word)
            if synsets:
                for synset in synsets:
                    for lemma in synset.lemmas():
                        lemma = lemma.name().lower()
                        if "_" not in lemma:
                            features['synonyms({})'.format(lemma)] = (lemma in document_words)
    return features


print(document_features(movie_reviews.words('pos/cv957_8737.txt')))

featuresets = [(document_features(d), c) for (d, c) in documents]
train_set, test_set = featuresets[100:], featuresets[:100]
classifier = NaiveBayesClassifier.train(train_set)

print(nltk.classify.accuracy(classifier, test_set))
classifier.show_most_informative_features(10)
