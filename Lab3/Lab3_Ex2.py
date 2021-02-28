from tweepy import OAuthHandler, API, Cursor
import nltk.data
from nltk.corpus.reader import CategorizedPlaintextCorpusReader, PlaintextCorpusReader
from nltk.tokenize.casual import TweetTokenizer
import re
from nltk.corpus import stopwords 
import sklearn 
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import confusion_matrix

# Exercise 5

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



# text wrangling functions:

def remove_emoji(string):  # github https://gist.github.com/slowkow/7a7f61f495e3dbb7e3d767f97bd7304b
    emoji_pattern = re.compile("["
                               u"\U0001F600-\U0001F64F"  # emoticons
                               u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                               u"\U0001F680-\U0001F6FF"  # transport & map symbols
                               u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                               u"\U00002500-\U00002BEF"  # chinese char
                               u"\U00002702-\U000027B0"
                               u"\U00002702-\U000027B0"
                               u"\U000024C2-\U0001F251"
                               u"\U0001f926-\U0001f937"
                               u"\U00010000-\U0010ffff"
                               u"\u2640-\u2642"
                               u"\u2600-\u2B55"
                               u"\u200d"
                               u"\u23cf"
                               u"\u23e9"
                               u"\u231a"
                               u"\ufe0f"  # dingbats
                               u"\u3030"
                               "]+", flags=re.UNICODE)
    return emoji_pattern.sub(r'', string)


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


def normalize_tweets(file, label):
    tweets = reader.raw(file).lower().split("\n")
    normalized_tweets = []
    for tweet in tweets:
        tweet = remove_links(tweet)
        #tweet = remove_emoji(tweet)
        tweet = remove_users(tweet)
        tweet = remove_numbers(tweet)
        tweet = remove_hashtags(tweet)
        tweet = [term for term in tweet.split(" ")
                                        if term not in stop_words]
        if tweet:
            normalized_tweets.append((" ".join(tweet), label))
    return normalized_tweets[:1600] # Barack Obama has more tweets, so I'm making it even 

tweets_with_labels = normalize_tweets(reader.fileids(categories="BarackObama"), 0) + normalize_tweets(reader.fileids(categories="NASA"), 1)

tweets = [tweet[0] for tweet in tweets_with_labels]
labels = [tweet[1] for tweet in tweets_with_labels]

tweets_train, tweets_test, labels_train, labels_test = train_test_split(tweets, labels, test_size=0.4, random_state=12) 

vectorizer = TfidfVectorizer()
tweets_train = vectorizer.fit_transform(tweets_train).todense() # todense() to satifsy input requirements

nb = GaussianNB()
nb.fit(tweets_train, labels_train)
tweets_test = vectorizer.transform(tweets_test).todense() # todense() to satifsy input requirements
label_prediction = nb.predict(tweets_test)

matrix = confusion_matrix(labels_test, label_prediction)

print(f"The probability of a tweet coming from Barack Obama is {round(((matrix[0][0] + matrix[0][1])/len(labels_test))*100, 2)}% (actual probaility: {round((matrix[0][0]/len(labels_test))*100, 2)}%)")
print(f"The probability of a tweet coming from NASA  is {round(((matrix[1][0] + matrix[1][1])/len(labels_test))*100, 2)}% (actual probability: {round((matrix[1][1]/len(labels_test))*100, 2)}%)")
print(f"The model has {round(nb.score(tweets_test, labels_test)*100, 2)}% accuracy")

