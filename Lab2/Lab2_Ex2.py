import nltk 
import sklearn
from nltk.corpus import brown
from nltk.corpus import nps_chat as chat
from nltk.tag import DefaultTagger, RegexpTagger, UnigramTagger, BigramTagger
from sklearn.model_selection import train_test_split

#a
splits = [[90,10], [50,50]]
correct_brown = brown.tagged_sents()
correct_chat = chat.tagged_posts()
default_tagger = DefaultTagger("NN")

for split in splits: #lag til funksjon for bruk i b
    test_brown, train_brown = train_test_split(correct_brown, test_size = split[1]/100, shuffle=False)
    test_chat, train_chat = train_test_split(correct_chat, test_size = split[1]/100, shuffle=False)

    default_tagger.tag(train_brown)
    print(f"The DefaultTagger accuracy for the Brown Corpus is {default_tagger.evaluate(test_brown)} using a {split[0]}/{split[1]} split.")
    default_tagger.tag(train_chat)
    print(f"The DefaultTagger accuracy for the NPS Chat Corpus is {default_tagger.evaluate(test_chat)} using a {split[0]}/{split[1]} split.\n")

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

regex_tagger = RegexpTagger(patterns, backoff=default_tagger)

for split in splits:
    test_brown, train_brown = train_test_split(correct_brown, test_size = split[1]/100, shuffle=False)
    test_chat, train_chat = train_test_split(correct_chat, test_size = split[1]/100, shuffle=False)
    
    # brown 

    print(f"--------- BROWN CORPUS TAGGING {split[0]}/{split[1]}---------\n")
    unigram_tagger_brown = UnigramTagger(train_brown, backoff=regex_tagger)
    bigram_tagger_brown = BigramTagger(train_brown, backoff=unigram_tagger_brown)

    print(f"The BigramTagger accuracy for the Brown Corpus is {bigram_tagger_brown.evaluate(test_brown)} using a {split[0]}/{split[1]} split.")
    print(f"The UnigramTagger accuracy for the Brown Corpus is {unigram_tagger_brown.evaluate(test_brown)} using a {split[0]}/{split[1]} split.")
    print(f"The RegexpTagger accuracy for the Brown Corpus is {regex_tagger.evaluate(test_brown)} using a {split[0]}/{split[1]} split.\n")     
    
    # chat

    print(f"--------- NPS CHAT CORPUS TAGGING {split[0]}/{split[1]}--------\n")
    unigram_tagger_chat = UnigramTagger(train_chat, backoff=regex_tagger)
    bigram_tagger_chat = BigramTagger(train_chat, backoff=unigram_tagger_chat)

    print(f"The BigramTagger accuracy for the NPS Chat Corpus is {bigram_tagger_chat.evaluate(test_chat)} using a {split[0]}/{split[1]} split.")
    print(f"The UnigramTagger accuracy for the NPS Chat Corpus is {unigram_tagger_chat.evaluate(test_chat)} using a {split[0]}/{split[1]} split.")
    print(f"The RegexpTagger accuracy for the NPS Chat Corpus is {regex_tagger.evaluate(test_chat)} using a {split[0]}/{split[1]} split.\n")




