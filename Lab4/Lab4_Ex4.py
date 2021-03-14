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


training_trump, padded_trump = padded_everygram_pipeline(5, tokenized_tweets)

trump_model = MLE(5)
trump_model.fit(training_trump, padded_trump)

# My very innovative solution (Google, hire me)

def predict_next(words): 
    predicted_words = words
    most_recent = ""
    while most_recent not in ["!", ".", "?", "<s>", "</s>"]: 
        maxscore = -1
        candidate_word = ""
        for word in list(trump_model.vocab):
            score = trump_model.score(word, predicted_words.split()[-4:]) # how likely is word x in context with the last three words? (trying to form a 4-gram)
            if score > maxscore: # find the most likely word in the given context
                maxscore = score
                candidate_word = word
        predicted_words += " " + candidate_word # when found, add to the existing context, and start over
        most_recent = candidate_word # to check if we've made a full sentence

    return predicted_words

print(predict_next("china is"))
 
 # Less innovative, though perhaps more accurate (?) and cleaner solution

def predict_next_actual(words):
    most_recent = words.split()[-1]
    if most_recent in ["<s>", "</s>"]:
        return " ".join(words.split()).replace(most_recent, "")
    elif most_recent in ["!", ".", "?"]:
        return " ".join(words.split())
    else:
        return predict_next_actual(words + " " + trump_model.generate(1, text_seed=words.split()))
    
print(predict_next_actual("china is"))