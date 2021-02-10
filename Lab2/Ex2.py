import nltk 
from nltk.corpus import brown
from nltk.corpus import nps_chat as chat
from nltk.tag import DefaultTagger, RegexpTagger, UnigramTagger, BigramTagger

#a
splits = [[90,10], [50,50]]
correct_brown = brown.tagged_sents()
correct_chat = chat.tagged_posts()
default_tagger = DefaultTagger("NN")

for split in splits: #lag til funksjon for bruk i b
    size_brown = int(len(correct_brown)*(split[0]/100))
    train_brown = correct_brown[:size_brown] #up to 90%
    test_brown = correct_brown[size_brown:] #from 90% to 100%

    size_chat = int(len(correct_chat)*(split[0]/100))
    train_chat = correct_chat[:size_chat]
    test_chat = correct_chat[size_chat:]

    default_tagger.tag(train_brown)
    print(f"The DefaultTagger accuracy for the Brown Corpus is {default_tagger.evaluate(test_brown)} using a {split[0]}/{split[1]} split.")
    default_tagger.tag(train_chat)
    print(f"The DefaultTagger accuracy for the NPS Chat Corpus is {default_tagger.evaluate(test_brown)} using a {split[0]}/{split[1]} split.")

    #50/50 is better because the tagger doesn't "learn", so when the test data is increased (from 10%) 
    #there's a bigger chance that some words are going to be NN? 

#b
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
regex_tagger = RegexpTagger(patterns)
unigram_tagger = UnigramTagger()
bigram_tagger = BigramTagger()
