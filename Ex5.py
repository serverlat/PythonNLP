from tweepy import OAuthHandler, API, Cursor
import nltk.data
from nltk.corpus.reader import CategorizedPlaintextCorpusReader, PlaintextCorpusReader
from nltk.tokenize.casual import TweetTokenizer
import re

# Exercise 5

consumer_key = "RMJSIt2Qmj4qnwXf04eIkjvru"
consumer_secret = "lUQh4rqpU4i1iU8n071niRBdrVLRBsh8hOtxuTUIVaQdEayHHW"
access_token = "2300132793-MD4A3wKlFmWUegL5vewodoUSWBFIHMlbSVUKigc"
access_token_secret = "UEMaOLNSUdeuEA5r2dcUuic30DgwwYdcLzuahgwPwMq4h"

# Boilerplate code from GitLab
auth = OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token, access_token_secret)
api = API(auth, wait_on_rate_limit=True)  # setting limit to avoid upsetting Twitter

topics = ["games", "food", "cats", "nature", "education", "computers", "love", "trump", "biden", "norway"]
for topic in topics:
    statuses = Cursor(api.search, q=f"{topic} -filter:retweets", tweet_mode="extended").items(200)
    for status in statuses:
        if status.lang == "en":
            file = open(f"C:/Users/olgur/natural_language_toolkit_data/twitter_corpus/tweets_{topic}.txt", "a",
                        encoding="utf-8")
            file.write(status.full_text)
            file.close()

reader = CategorizedPlaintextCorpusReader("C:/Users/olgur/natural_language_toolkit_data/twitter_corpus",
                                          r'tweets_.*\.txt', cat_pattern=r'tweets_(\w+)\.txt')

# setting up stopwords
stopword_reader = PlaintextCorpusReader("C:/Users/olgur/natural_language_toolkit_data/twitter_corpus/twitterstopwords/",
                                        r'.*\.txt', encoding='latin-1')
stop_words = set(['“', '”', '’', ",", "#", "—", "__", "_", "___"])

for file in stopword_reader.fileids():
    stops = stopword_reader.raw(file).replace("\n", ",").split(",")
    for word in stops:
        stop_words.add(word)


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
    http_regex = re.compile(r"https://.*")
    return http_regex.sub(r"", text)


def remove_users(text):
    user_regex = re.compile(r"@.*")
    return user_regex.sub(r"", text)


def remove_numebrs(text):
    number_regex = re.compile(r"[0-9]+")
    return number_regex.sub(r"", text)


def remove_hashtags(text):
    return re.sub(r"#", r"", text)


# a
def tokenize_tweets(file):
    tweet_tokenizer = TweetTokenizer()
    text = reader.raw(file)
    link_free = remove_links(text)
    emoji_free = remove_emoji(link_free)
    user_free = remove_users(emoji_free)
    number_free = remove_numebrs(user_free)
    hashtag_free = remove_hashtags(number_free)
    twitter_words = [term.lower() for term in tweet_tokenizer.tokenize(hashtag_free)
                     if term.lower() not in stop_words]
    twitter_words_with_hashtags = [term.lower() for term in tweet_tokenizer.tokenize(number_free) if
                                   term.lower() not in stop_words]
    return twitter_words, twitter_words_with_hashtags


corpus_tokens = []

for category in reader.categories():
    for file in reader.fileids(categories=category):
        without_hashtags, with_hashtags = tokenize_tweets(file)

        # c
        fdist_category = nltk.FreqDist(without_hashtags)
        print("Most common words in", category, ":", fdist_category.most_common(10))

        # d
        hashtags = [word for word in with_hashtags if word.startswith("#")]
        fdist_category_hashtag = nltk.FreqDist(hashtags)
        print("Most common hashtags in", category, ":", fdist_category_hashtag.most_common(10))

        corpus_tokens += without_hashtags

fdist = nltk.FreqDist(corpus_tokens)
# b
print(fdist.most_common(10))
