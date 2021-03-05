from tweepy import OAuthHandler, API, Cursor
import nltk.data
from nltk.corpus.reader import CategorizedPlaintextCorpusReader, PlaintextCorpusReader
from nltk.tokenize.casual import TweetTokenizer
import re
from nltk.corpus import stopwords 
import sklearn 
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import confusion_matrix


# Exercise 3

consumer_key = "RMJSIt2Qmj4qnwXf04eIkjvru"
consumer_secret = "lUQh4rqpU4i1iU8n071niRBdrVLRBsh8hOtxuTUIVaQdEayHHW"
access_token = "2300132793-MD4A3wKlFmWUegL5vewodoUSWBFIHMlbSVUKigc"
access_token_secret = "UEMaOLNSUdeuEA5r2dcUuic30DgwwYdcLzuahgwPwMq4h"

# Boilerplate code from GitLab
auth = OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token, access_token_secret)
api = API(auth, wait_on_rate_limit=True)  # setting limit to avoid upsetting Twitter

'''accounts = [("NASA", 11348282), ("BarackObama", 813286)]
for account in accounts:
    statuses = Cursor(api.user_timeline, user_id=account[1], include_rts=False, exclude_replies=True, count=10000, tweet_mode="extended").items()
    for status in statuses:
        if status.lang == "en":
            file = open(f"C:/Users/olgur/nltk_data/twitter_corpus/tweets_{account[0]}.txt", "a",
                        encoding="utf-8")
            file.write(status.full_text.replace("\n", " ") + "\n")
            file.close()'''

reader = CategorizedPlaintextCorpusReader("C:/Users/olgur/nltk_data/twitter_corpus",
                                          r'tweets_.*\.txt', cat_pattern=r'tweets_(\w+)\.txt')

# setting up stopwords
stop_words = set(['“', '”', '’', ",", "#", "—", "__", "_", "___", ".", ":", '"', "?", "!", "-", ")", "(", "...", "$"]).union(set(stopwords.words("english")))

def remove_links(text):
    http_regex = re.compile(r"(https|http)://.*")
    return http_regex.sub(r"", text)


def remove_users(text):
    user_regex = re.compile(r"@.*")
    return user_regex.sub(r"", text)


def remove_numbers(text):
    number_regex = re.compile(r"[0-9]+")
    return number_regex.sub(r"", text)


def remove_hashtags(text):
    return re.sub(r"#", r"", text)

prob_test_tweets = [[],[]]

def normalize_tweets(file, label):
    tweets = reader.raw(file).lower().split("\n")
    normalized_tweets = []
    counter = 0
    original = ""
    for tweet in tweets:
        original = tweet
        tweet = remove_links(tweet)
        tweet = remove_users(tweet)
        tweet = remove_numbers(tweet)
        tweet = remove_hashtags(tweet)
        tweet = [term for term in tweet.split(" ")
                                        if term not in stop_words]
        if tweet:
            if counter < 5: 
                prob_test_tweets[label].append((" ".join(tweet), original))
                counter +=1
            else: 
                normalized_tweets.append((" ".join(tweet), label))
    return normalized_tweets[:1600] # Barack Obama has more tweets, so I'm making it even 

tweets_with_labels = normalize_tweets(reader.fileids(categories="BarackObama"), 0) + normalize_tweets(reader.fileids(categories="NASA"), 1)

tweets = [tweet[0] for tweet in tweets_with_labels]
labels = [tweet[1] for tweet in tweets_with_labels]

tweets_train, tweets_test, labels_train, labels_test = train_test_split(tweets, labels, test_size=0.1, random_state=12) 

vectorizer = TfidfVectorizer()
tweets_train = vectorizer.fit_transform(tweets_train) 

nb = MultinomialNB()
nb.fit(tweets_train, labels_train)
tweets_test = vectorizer.transform(tweets_test) 
label_prediction = nb.predict(tweets_test)

matrix = confusion_matrix(labels_test, label_prediction)

print(f"The model has an overall accuracy of {round(nb.score(tweets_test, labels_test)*100, 2)}% \n")

print("\n-------------- BARACK OBAMA TEST TWEETS ------------\n")
for tweet in prob_test_tweets[0]:
    vector = vectorizer.transform([tweet[0]])
    print(f"The tweet '{tweet[1]}' has a likelyhood of {round(nb.predict_proba(vector)[0][0],2)} to be tweeted by Barack Obama and not NASA\n")

print("\n--------------- NASA TEST TWEETS --------------------\n")
for tweet in prob_test_tweets[1]:
    vector = vectorizer.transform([tweet[0]])
    print(f"The tweet '{tweet[1]}' has a likelyhood of {round(nb.predict_proba(vector)[0][1],2)} to be tweeted by NASA and not Barack Obama\n")

