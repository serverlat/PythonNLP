import nltk 
from nltk.corpus import names
from nltk.classify import NaiveBayesClassifier, DecisionTreeClassifier, MaxentClassifier, accuracy
import random
import sklearn
from sklearn.model_selection import train_test_split

male_names = [(name, "m") for name in names.words("male.txt")]
female_names = [(name, "f") for name in names.words("female.txt")]
all_names = male_names + female_names
random.shuffle(all_names)

def gender_feature(name):
    return {"last_letter": name[-1]}

feature_sets = [(gender_feature(name), gender) for (name, gender) in all_names]
train_features, test_features = train_test_split(feature_sets, test_size=0.1)

# Decision Tree
dec_tree = DecisionTreeClassifier.train(train_features)
print(f"The accuracy of the Decision Tree classifier is {accuracy(dec_tree, test_features)}")

 # Naive Bayes
nb = NaiveBayesClassifier.train(train_features)
print(f"The accuracy of the Naive Bayes classifier is {accuracy(nb, test_features)}")

 # Maximum Entropy
max_ent = MaxentClassifier.train(train_features, trace=0) # set trace to 0 to not print the log
print(f"The accuracy of the Maximum Entropy classifier is {accuracy(max_ent,test_features)}")

