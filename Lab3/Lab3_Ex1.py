import nltk 
from nltk.corpus import names
from nltk.classify import NaiveBayesClassifier, DecisionTreeClassifier, MaxentClassifier, accuracy
import random
import sklearn
from sklearn.model_selection import train_test_split
from nltk import ConfusionMatrix

male_names = [(name, "m") for name in names.words("male.txt")]
female_names = [(name, "f") for name in names.words("female.txt")]
all_names = male_names + female_names

def gender_feature(name):
    return {"last_letter": name[-1]}

feature_sets = [(gender_feature(name), gender) for (name, gender) in all_names]
train_features, test_features = train_test_split(feature_sets, test_size=0.1)

def performance(classifier):
    test = []
    gold = []

    for i in range(len(test_features)):
        test.append(classifier.classify(test_features[i][0]))
        gold.append(test_features[i][1])

    matrix = ConfusionMatrix(gold, test)
    labels = {"female", "male"}

    tp = matrix["f","f"]
    fn = matrix["m", "f"]
    fp = matrix["f", "m"]
    tn = matrix["m", "m"]

    precision_female = tp/(tp + fp)
    precision_male = tn/(tn + fn)
    recall_female = tp/(tp + fn) # actual female/ actual female + missed female
    recall_male = tn/(tn + fp)
    fscore_female = 2 * precision_female * recall_female / precision_female + recall_female
    fscore_male = 2 * precision_male * recall_male / precision_male + recall_male

    print(f"Precision for female names: {round(precision_female,2)}")
    print(f"Precision for male names: {round(precision_male,2)}")
    print(f"Recall for female names: {round(recall_female,2)}")
    print(f"Recall for male names: {round(recall_male,2)}")
    print(f"F-score for female names: {round(fscore_female,2)}")
    print(f"F-score for male names: {round(fscore_male,2)}")
    print("\n")

# Decision Tree
dec_tree = DecisionTreeClassifier.train(train_features)
print(f"The accuracy of the Decision Tree classifier is {round(accuracy(dec_tree, test_features),2)}")
performance(dec_tree)

 # Naive Bayes
nb = NaiveBayesClassifier.train(train_features)
print(f"The accuracy of the Naive Bayes classifier is {round(accuracy(nb, test_features),2)}")
performance(nb)

 # Maximum Entropy
max_ent = MaxentClassifier.train(train_features, trace=0) # set trace to 0 to not print the log
print(f"The accuracy of the Maximum Entropy classifier is {round(accuracy(max_ent, test_features),2)}")
performance(max_ent)
