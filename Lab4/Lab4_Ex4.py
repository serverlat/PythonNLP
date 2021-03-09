import nltk
import json
import re
from nltk.lm.api import LanguageModel
from nltk.util import ngrams
from nltk import sent_tokenize, word_tokenize
from nltk.lm.preprocessing import padded_everygram_pipeline
from nltk.lm import MLE

tokenized_tweets = []

with open('C:/Users/olgur/Programming/PythonNLP/Lab4/realDonaldTrump.json', encoding="utf8") as f:
    tweets = json.load(f)
    for data in tweets:
        tweet_sents = sent_tokenize(data["text"].lower())
        tweet_words = [word_tokenize(sentence) for sentence in tweet_sents]
        for words in tweet_words: tokenized_tweets.append(words)


training_trump, padded_trump = padded_everygram_pipeline(4, tokenized_tweets)

trump_model = MLE(3)
trump_model.fit(training_trump, padded_trump)

def predict_next(words): #trigram only
    predicted_words = words
    most_recent = ""
    while most_recent not in ["!", ".", "?", "<s>"]:
        maxscore = -1
        candidate_word = ""
        for word in list(trump_model.vocab):
            score = trump_model.score(word, predicted_words.split()[-2:])
            if score > maxscore:
                maxscore = score
                candidate_word = word
        predicted_words += " " + candidate_word
        most_recent = candidate_word

    print(predicted_words)

predict_next("america is")







def clean_data(data):
    clean_data = []
    for text in data:
        text = re.sub(r"(https|http)://.*", r"", text) # most links
        text = re.sub(r"@.*", r"", text) # user mentions
        text = re.sub (r"[0-9]+", r"", text) # numbers
        text = text.replace("\\n", "")
        text = re.sub(r"[^\w\s]", r"", text) # punctuations, also supposed to remove \n etc. but it doesn't work
        text = re.sub(r"\s+", r" ", text) # duplcate space removal
        #text = [unidecode.unidecode(word.lower()) for word in text.split() if word not in stop_words] # accent characters
        #text = [lemmatizer.lemmatize(word) for word in text] # lemmatization
        text = " ".join(text)
        clean_data.append(text)
    return clean_data