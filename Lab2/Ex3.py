import nltk 
from nltk.tag import UnigramTagger, BigramTagger, RegexpTagger
from nltk.corpus import brown
from nltk.corpus import nps_chat as chat

#a

# LookupTagger setup from NLTK Chapter 5

fdist = nltk.FreqDist(brown.words())
cfdist = nltk.ConditionalFreqDist(brown.tagged_words())
top_words = fdist.most_common(200)
most_likely_tags = dict((word, cfdist[word].max()) for (word, _) in top_words) 
default_tagger = UnigramTagger(model=most_likely_tags)

splits = [[90,10], [50,50]]
correct_brown = brown.tagged_sents()[:int(len(brown.tagged_sents())*0.25)]
correct_chat = chat.tagged_posts()[:int(len(chat.tagged_posts())*0.25)]

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

regex_tagger = RegexpTagger(patterns, backoff=default_tagger)

accuracies = []

for split in splits:
    size_brown = int(len(correct_brown)*(split[0]/100))
    train_brown = correct_brown[:size_brown] #up to 90%
    test_brown = correct_brown[size_brown:] #from 90% to 100%

    size_chat = int(len(correct_chat)*(split[0]/100))
    train_chat = correct_chat[:size_chat]
    test_chat = correct_chat[size_chat:]
    
    # brown
    unigram_tagger_brown = UnigramTagger(train_brown, backoff=regex_tagger)
    bigram_tagger_brown = BigramTagger(train_brown, backoff=unigram_tagger_brown)

    print(f"--------- BROWN CORPUS TAGGING {split[0]}/{split[1]}---------\n")
    print(f"The BigramTagger accuracy for the Brown Corpus is {bigram_tagger_brown.evaluate(test_brown)}")
    print(f"The UnigramTagger accuracy for the Brown Corpus is {unigram_tagger_brown.evaluate(test_brown)}")
    print(f"The RegexpTagger accuracy for the Brown Corpus is {regex_tagger.evaluate(test_brown)}")  
    print(f"The LookupTagger accuracy for the Brown Corpus is {default_tagger.evaluate(test_brown)}\n")   
    
    #chat
    unigram_tagger_chat = UnigramTagger(train_chat, backoff=regex_tagger)
    bigram_tagger_chat = BigramTagger(train_chat, backoff=unigram_tagger_chat)

    print(f"--------- NPS CHAT CORPUS TAGGING {split[0]}/{split[1]}--------\n")
    print(f"The BigramTagger accuracy for the NPS Chat Corpus is {bigram_tagger_chat.evaluate(test_chat)}")
    print(f"The UnigramTagger accuracy for the NPS Chat Corpus is {unigram_tagger_chat.evaluate(test_chat)}")
    print(f"The RegexpTagger accuracy for the NPS Chat Corpus is {regex_tagger.evaluate(test_chat)}")
    print(f"The LookupTagger accuracy for the NPS Chat Corpus is {default_tagger.evaluate(test_chat)}\n")

#b 

