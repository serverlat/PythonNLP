import pandas as pd
import pickle
import numpy as np
import cmudict
from models import InferSent
from wonderlic_nlp import WonderlicNLP 
from collections import defaultdict
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LinearRegression

cmudict = cmudict.dict()
wnlp = WonderlicNLP()
embeddings_index = {}

def initialize_glove():
    f = open('LCP/glove.42B.300d.txt', encoding="utf8")
    for line in f: # fill dictionary with data from glove for later use
        values = line.split()
        word = values[0]
        coefs = np.asarray(values[1:], dtype='float32') # values for the word
        embeddings_index[word] = coefs
    f.close()

initialize_glove()
train_data = pd.read_csv("LCP/CompLex/train/lcp_single_train.tsv", sep="\t")

def glove_embedding(word):
    return [emb for emb in embeddings_index[str(word).lower()]]

def extract_features(tokens):
    features = defaultdict(list)
    for token in tokens:
        token = str(token).lower()
        mrc_features = wnlp.get_mrc_features(token)
        glove = glove_embedding(token)
        for i in range(1,301):
            features[f"glove{i}"].append(glove[i-1])
        features["word_length"].append(mrc_features["Nlet"])
        features["syl_count"].append(mrc_features["Nsyl"])
        features["brown_freq"].append(mrc_features["Brown-freq"])
        features["familiarity"].append(mrc_features["Fam"])
        features["concreteness"].append(mrc_features["Conc"])
        features["imagability"].append(mrc_features["Imag"])
        features["meaningfulness_c"].append(mrc_features["Meanc"])
        features["meaningfulness_p"].append(mrc_features["Meanp"])
        features["age_of_aquisition"].append(mrc_features["AOA"])
    return features


features_df = pd.DataFrame(extract_features(train_data["token"].values))
train_labels = train_data["complexity"]
test_data = pd.read_csv("LCP/CompLex/test/lcp_single_test.tsv", sep="\t")

#test_data["token_en"] = test_data["token"].apply(glove_embedding)
#test_text =  test_data["token_en"]
test_labels = pd.read_csv("LCP/CompLex/test-labels/lcp_single_test.tsv", sep="\t")["complexity"]

model = LinearRegression().fit(features_df, train_labels)
print(model.score(features_df, train_labels))
#labels_pred = model.predict(test_text)
#for i in range(len(labels_pred)):
    #print("Predicted: ", labels_pred[i], end="")
    #print(" Actual: ", test_labels[i])
