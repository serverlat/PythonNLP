from collections import Counter
from textblob import TextBlob
from datetime import datetime
import pandas as pd
import numpy as np
import re
import matplotlib.pyplot as plt
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Dense, Embedding, GRU, Dropout, LSTM
from sklearn.model_selection import train_test_split
import collections

MAX_WORDS = 1000

def sentiment_analysis(sentence):
    analysis = TextBlob(sentence)
    if analysis.sentiment.polarity > 0:
        return 'pos'
    elif analysis.sentiment.polarity == 0:
        return 'neu'
    else:
        return 'neg'

def load_tweets():
    tweets = pd.read_json("Lab5/realDonaldTrump.json")
    tweets["sentiment"] = np.array([sentiment_analysis(tweet)
                        for tweet in tweets["text"]])
    print(len(tweets))
    return tweets

def sequences():
    tweets = load_tweets().sort_values(by="timeStamp")
    tokenizer = Tokenizer(num_words=MAX_WORDS, split=" ")
    tokenizer.fit_on_texts(tweets["text"].values) # fills internal vocab with unique number for each word, lower value = more frequent {the: 1, cat:2 ...}
    election_tweets = tweets[(tweets["timeStamp"] < "2020-12-03 00:00:00") & (tweets["timeStamp"] > "2020-10-03 00:00:00")]
    regular_tweets = tweets[(tweets["timeStamp"] < "2020-10-03 00:00:00")]
    election_tweets_temp = []
    election_tweets_dates = []
    for date, group in election_tweets.groupby(pd.Grouper(key="timeStamp", freq="D")):
        election_tweets_dates.append(date)
        election_tweets_temp.append(pad_sequences(tokenizer.texts_to_sequences(group["text"].values), 50))
    tweets_labels = pd.get_dummies(regular_tweets["sentiment"]).values # converts one column with x possible values into x separate columns with true/false, like [sentiment = pos, neu or neg] -> [pos = 0/1, neu = 0/1, neg = 0/1]
    regular_tweets = tokenizer.texts_to_sequences(regular_tweets["text"].values) # converts texts (list of sentences) to lists of numbers from the vocab like [1, 2, 34, 221, 4422]
    regular_tweets = pad_sequences(regular_tweets, 50) # pads sequences so that they're all the same length, pads with 0 (reserved for padding)
    return election_tweets_dates, election_tweets_temp, regular_tweets, tweets_labels, tokenizer

election_tweets_dates, election_tweets, tweets_text, tweets_labels, tokenizer = sequences()

def embedding(tokenizer):
    word_index = tokenizer.word_index
    embeddings_index = {}
    f = open('Lab5/glove.twitter.27B.100d.txt', encoding="utf8")
    for line in f: # fill dictionary with data from glove for later use
        values = line.split()
        word = values[0]
        coefs = np.asarray(values[1:], dtype='float32') # values for the word
        embeddings_index[word] = coefs
    f.close()

    print('Found %s word vectors.' % len(embeddings_index))
    embedding_matrix = np.zeros((len(word_index) + 1, 100)) # emtpy array with as many items as the embedding index vocab x glove value length
    for word, value in word_index.items():
        embedding_vector = embeddings_index.get(word) # get glove values for the word from corpus
        if embedding_vector is not None:
            embedding_matrix[value] = embedding_vector # {value for word (index): values for word from glove}
    # Embedding layer turns positive integers (indexes) (from fit_on_texts) into dense vectors of fixed size (from glove)
    return Embedding(len(word_index) + 1, # vocab size
                     100, # dimension of the embedding
                     weights=[embedding_matrix], # map input integers to embedding vectors, so if we input a sentence [121 213 321 11 22 44] it can be mapped word for word
                     input_length=50, #trying 25
                     mask_zero=True,
                     trainable=False)

def buildModel(embedding_layer):
    model = Sequential()
    model.add(embedding_layer)
    model.add(LSTM(40, dropout=0.2))
    model.add(Dropout(0.2))
    model.add(Dense(3, activation='softmax'))
    model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])
    return model

model = buildModel(embedding(tokenizer))
print(model.summary())

def processing(model, X, Y):
    X_train, X_test, Y_train, Y_test = train_test_split(
        X, Y, test_size=0.1)
    history = model.fit(X_train, Y_train, epochs=10,
              batch_size=32, verbose=1, callbacks=None,)
    scores = model.evaluate(X_test, Y_test, verbose=1)
    print(scores)
    print("Accuracy: %.2f" % (scores[1]))


processing(model, tweets_text, tweets_labels)

tweet_sents = Counter(np.argmax(model.predict(election_tweets[0]), axis=-1))
print(tweet_sents.items())

plot_vals_positive = [tweet_sents[2]]
plot_vals_negative = [tweet_sents[0]]
plot_vals_neutral = [tweet_sents[1]]

for i in range(1, len(election_tweets_dates)):
    tweet_sents = Counter(np.argmax(model.predict(election_tweets[i]), axis=-1))

    plot_vals_positive.append(plot_vals_positive[i-1])
    plot_vals_negative.append(plot_vals_negative[i-1])
    plot_vals_neutral.append(plot_vals_neutral[i-1])

    plot_vals_positive[i] = plot_vals_positive[i-1] + tweet_sents[2]
    plot_vals_neutral[i] = plot_vals_neutral[i-1] + tweet_sents[1]
    plot_vals_negative[i] = plot_vals_negative[i-1] + tweet_sents[0]

plt.plot(election_tweets_dates, plot_vals_positive, label = "positive") 
plt.plot(election_tweets_dates, plot_vals_negative, label = "negative") 
plt.plot(election_tweets_dates, plot_vals_neutral, label = "neutral")
plt.xticks(election_tweets_dates, [f"{date.date().month}/{date.date().day}" for date in election_tweets_dates], rotation=90)
plt.legend()    
plt.show()