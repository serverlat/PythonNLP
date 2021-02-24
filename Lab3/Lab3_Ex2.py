from tweepy import OAuthHandler, API, Cursor
import nltk.data
from nltk.corpus.reader import CategorizedPlaintextCorpusReader, PlaintextCorpusReader
from nltk.tokenize.casual import TweetTokenizer
import re
from nltk.corpus import stopwords 
import sklearn 
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split

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
    statuses = Cursor(api.user_timeline, user_id=account[1], include_rts=False, exclude_replies=True, count=3000, tweet_mode="extended").items()
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


def tokenize_tweets(file):
    #tweet_tokenizer = TweetTokenizer(strip_handles=True)
    tweets = reader.raw(file).lower().split("\n")
    normalized_tweets = []
    for tweet in tweets:
        link_free = remove_links(tweet)
        emoji_free = remove_emoji(link_free)
        user_free = remove_users(emoji_free)
        number_free = remove_numbers(user_free)
        hashtag_free = remove_hashtags(number_free)
        tokenized_tweet = [term for term in hashtag_free.split(" ")
                                        if term not in stop_words]
        if tokenized_tweet:
            normalized_tweets.append((" ".join(tokenized_tweet), file))
    return normalized_tweets

tweets_with_labels = tokenize_tweets(reader.fileids(categories="BarackObama")) + tokenize_tweets(reader.fileids(categories="NASA"))

tweets = [tweet[0] for tweet in tweets_with_labels]
labels = [tweet[1] for tweet in tweets_with_labels]

nasa_train, nasa_test, obama_train, obama_test = train_test_split(tweets, labels, test_size=0.1, random_state=12)

def vectorize_tweets(tweets):
    vectorizer = TfidfVectorizer()
    return vectorizer.fit_transform(tweets)



