import nltk
from nltk.corpus import state_union
from nltk.corpus import cmudict as d
import re
from selenium import webdriver
from webdriver_manager.chrome import ChromeDriverManager
from bs4 import BeautifulSoup
from tweepy import OAuthHandler, API, Cursor
import nltk.data
from nltk.corpus.reader import CategorizedPlaintextCorpusReader, PlaintextCorpusReader
from nltk.tokenize.casual import TweetTokenizer

# Exercise 1
words = ['she', 'sells', 'sea', 'shells', 'by', 'the', 'sea', 'shore']

# a
print("Words that start with 'sh':")
for word in words:
    if word[:2] == "sh":
        print(word, end=" ")
print("\n")

# b
print("Words longer than 4 characters:")
for word in words:
    if len(word) > 4:
        print(word, end=" ")
print("\n")

# Exercise 2

# a
files = list(state_union.fileids())
terms = ["men", "women", "people"]
statistics = nltk.ConditionalFreqDist((file, word)
                                      for file in state_union.fileids()
                                      for word in state_union.words(file)
                                      for term in terms if word.lower() == term)
statistics.tabulate(conditions=files, samples=terms)

# b
years_raw = sorted(list(set([int(year[:4]) for year in state_union.fileids()])))
years = [str(year) for year in years_raw]
year_statistics = nltk.ConditionalFreqDist((word.lower(), fileid[:4])
                                           for fileid in state_union.fileids()
                                           for word in state_union.words(fileid)
                                           for term in terms
                                           if word.lower() == term)
year_statistics.plot()
# More women over time, a lot of people in 1995 and 1946, more or less stable amount of men.

# Exercise 3
dictionary = d.dict()  # the CMU pronouncing dictionary


# syllables function borrowed from:
# https://stackoverflow.com/questions/5876040/number-of-syllables-for-words-in-a-text
def syllables(word):
    if word in dictionary:
        return max([len([syl for syl in entry if syl[-1].isdigit()]) for entry in dictionary[
            word.lower()]])
    else:
        return -1
        # the number of stressed phonems in a word is approximately the number of syllables
        # makes lists of syllables, takes max if there are more than one pronunciation
        # the number of syllables is approximately the number of


# a doesn't work for vowel sounds like honest
def word_to_pig_latin(word):
    piglatinized = ""
    vowels = [index for index, char in enumerate(word) if char.lower() in "aeiou"]
    if word[0].lower() in "aeiou":
        if syllables(word) > 1:
            piglatinized = word[vowels[1]:] + word[:vowels[1]] + "ay"
        else:
            piglatinized = word + "yay"
    else:
        if word[1] in "aeiou":
            piglatinized = word[1:] + word[0] + "ay"
        else:
            if len(word) > 2:
                piglatinized = word[vowels[0]:] + word[:vowels[0]] + "ay"
            else:
                piglatinized = word[-1] + word[:0] + "ay"
    return piglatinized


# b
# not accounting for apostrophes, the word "am" doesnt work
def text_to_pig_latin(text):
    pigtext = ""
    char_indices = [index for index in range(len(text)) if text[index].isalpha() is False]
    tokens = nltk.word_tokenize(text)
    counter = 0
    for word in tokens:
        if word.isalpha():
            newword = word_to_pig_latin(word.lower())
            if word[0].isupper():
                newword = newword[0].upper() + newword[1:]
            pigtext += newword
            counter += len(word)
            while counter in char_indices:
                pigtext += text[counter]
                counter += 1
    return pigtext


# c
# First we have to remove all the "ay" and "yay" from the words. For words than begin with vowels, we're done
# after having removed "yay" (unless the alternative encoding is used, where the first vowel + consonant cluster is
# removed). Then we have to distinguish between words beginning with one consonant or a consonant cluster. The only
# way I can think of is to try both cases (i.e. moving the last letter to the first place and moving the two last
# letters to the first place) and checking if any of the words make sense, which would be very time consuming, at least
# if we're using Wikipedia's definitions of Pig Latin.
# Ex. keyboard -> erboardkay and training -> ainingtray would become either keyboard/dkeyboar and rainingt/training
# and the computer wouldn't know any better which one is correct. We'd also have to account for words that start with
# three consonants, which would add even more ambiguity to the task. I don't know how to do this without human
# intervention or some kind of thesaurus that's available for the computer to check.

# Exercise 4
driver = webdriver.Chrome(ChromeDriverManager().install())
driver.get("https://old.reddit.com/")
content = driver.page_source
soup = BeautifulSoup(content, features="html.parser")
tabs = soup.find_all("div", class_="linkflair")

for tab in tabs:
    subreddit = tab.select("a.subreddit")[0].get_text()
    upvotes = tab.select("div.score.unvoted")[0].get_text()
    post_title = tab.select("p.title a.title")[0].get_text()
    time = tab.select("time")[0]["title"]
    print(subreddit, time, upvotes, post_title)

driver.quit()

# Exercise 5

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
