import nltk 
import sklearn
from nltk.tag import UnigramTagger, BigramTagger, RegexpTagger
from nltk.corpus import brown
from nltk.corpus import nps_chat as chat
from sklearn.model_selection import train_test_split

#a

# LookupTagger setup from NLTK Chapter 5
# For Brown Corpus

fdist_brown = nltk.FreqDist(brown.words()[:int((len(brown.words())-1))])  # slicing to vary the size of the dataset
cfdist_brown = nltk.ConditionalFreqDist(brown.tagged_words())
top_words_brown = fdist_brown.most_common(200)
most_likely_tags_brown = dict((word, cfdist_brown[word].max()) for (word, _) in top_words_brown) 
default_tagger_brown = UnigramTagger(model=most_likely_tags_brown)

splits = [[90,10], [50,50]]
correct_brown = brown.tagged_sents()[:int((len(brown.tagged_sents())-1))] # slicing to vary the size of the dataset
correct_chat = chat.tagged_posts()[:int((len(chat.tagged_posts())-1))]

patterns = [
     (r'.*ing$', 'VBG'),                # gerunds
     (r'.*ed$', 'VBD'),                 # simple past
     (r'.*es$', 'VBZ'),                 # 3rd singular present
     (r'.*ould$', 'MD'),                # modals
     (r'.*\'s$', 'NN$'),                # possessive nouns
     (r'.*s$', 'NNS'),                  # plural nouns
     (r'^-?[0-9]+(\.[0-9]+)?$', 'CD'),  # cardinal numbers
     (r'.*', 'NN')                      # nouns (default)
]


for split in splits:
    test_brown, train_brown = train_test_split(correct_brown, test_size = split[1]/100, shuffle=False)
    test_chat, train_chat = train_test_split(correct_chat, test_size = split[1]/100, shuffle=False)
    
    # brown
    regex_tagger_brown = RegexpTagger(patterns, backoff=default_tagger_brown)
    unigram_tagger_brown = UnigramTagger(train_brown, backoff=regex_tagger_brown)
    bigram_tagger_brown = BigramTagger(train_brown, backoff=unigram_tagger_brown)

    print(f"--------- BROWN CORPUS TAGGING {split[0]}/{split[1]}---------\n")
    print(f"The BigramTagger accuracy for the Brown Corpus is {round(bigram_tagger_brown.evaluate(test_brown),3)}")
    print(f"The UnigramTagger accuracy for the Brown Corpus is {round(unigram_tagger_brown.evaluate(test_brown),3)}")
    print(f"The RegexpTagger accuracy for the Brown Corpus is {round(regex_tagger_brown.evaluate(test_brown),3)}")  
    print(f"The LookupTagger accuracy for the Brown Corpus is {round(default_tagger_brown.evaluate(test_brown),3)}\n")   
    
    #chat
    regex_tagger_chat = RegexpTagger(patterns, backoff=default_tagger_brown)
    unigram_tagger_chat = UnigramTagger(train_chat, backoff=regex_tagger_chat)
    bigram_tagger_chat = BigramTagger(train_chat, backoff=unigram_tagger_chat)

    print(f"--------- NPS CHAT CORPUS TAGGING {split[0]}/{split[1]}--------\n")
    print(f"The BigramTagger accuracy for the NPS Chat Corpus is {round(bigram_tagger_chat.evaluate(test_chat),3)}")
    print(f"The UnigramTagger accuracy for the NPS Chat Corpus is {round(unigram_tagger_chat.evaluate(test_chat),3)}")
    print(f"The RegexpTagger accuracy for the NPS Chat Corpus is {round(regex_tagger_chat.evaluate(test_chat),3)}")
    print(f"The LookupTagger accuracy for the NPS Chat Corpus is {round(default_tagger_brown.evaluate(test_chat),3)}\n")

#b 

