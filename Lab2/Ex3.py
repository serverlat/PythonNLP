import nltk 
from nltk.tag import UnigramTagger, BigramTagger, RegexpTagger
from nltk.corpus import brown
from nltk.corpus import nps_chat as chat

#a

# LookupTagger setup from NLTK Chapter 5
# For Brown Corpus

fdist_brown = nltk.FreqDist(brown.words()) # change this to fit the size? but idk how 
cfdist_brown = nltk.ConditionalFreqDist(brown.tagged_words())
top_words_brown = fdist_brown.most_common(200)
most_likely_tags_brown = dict((word, cfdist_brown[word].max()) for (word, _) in top_words_brown) 
default_tagger_brown = UnigramTagger(model=most_likely_tags_brown)

# For NPS chat 

fdist_chat = nltk.FreqDist(chat.words())
cfdist_chat = nltk.ConditionalFreqDist(chat.tagged_words())
top_words_chat = fdist_chat.most_common(200)
most_likely_tags_chat = dict((word, cfdist_chat[word].max()) for (word, _) in top_words_chat)
default_tagger_chat = UnigramTagger(model=most_likely_tags_chat) 


splits = [[90,10], [50,50]]
correct_brown = brown.tagged_sents()[:int(len(brown.tagged_sents()))]
correct_chat = chat.tagged_posts()[:int(len(chat.tagged_posts()))]

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
    size_brown = int(len(correct_brown)*(split[0]/100))
    train_brown = correct_brown[:size_brown] #up to 90%
    test_brown = correct_brown[size_brown:] #from 90% to 100%

    size_chat = int(len(correct_chat)*(split[0]/100))
    train_chat = correct_chat[:size_chat]
    test_chat = correct_chat[size_chat:]
    
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
    regex_tagger_chat = RegexpTagger(patterns, backoff=default_tagger_chat)
    unigram_tagger_chat = UnigramTagger(train_chat, backoff=regex_tagger_chat)
    bigram_tagger_chat = BigramTagger(train_chat, backoff=unigram_tagger_chat)

    print(f"--------- NPS CHAT CORPUS TAGGING {split[0]}/{split[1]}--------\n")
    print(f"The BigramTagger accuracy for the NPS Chat Corpus is {round(bigram_tagger_chat.evaluate(test_chat),3)}")
    print(f"The UnigramTagger accuracy for the NPS Chat Corpus is {round(unigram_tagger_chat.evaluate(test_chat),3)}")
    print(f"The RegexpTagger accuracy for the NPS Chat Corpus is {round(regex_tagger_chat.evaluate(test_chat),3)}")
    print(f"The LookupTagger accuracy for the NPS Chat Corpus is {round(default_tagger_chat.evaluate(test_chat),3)}\n")

#b 

