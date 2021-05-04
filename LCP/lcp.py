import pandas as pd
import pickle
import numpy as np
import cmudict
import torch 
import math
import csv
import nltk
import re
from datetime import datetime
from nltk.stem import WordNetLemmatizer
from wonderlic_nlp import WonderlicNLP 
from collections import defaultdict
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from tqdm.auto import tqdm
from matplotlib import pyplot as plt

class LCP:

    def __init__(self):
        self.filename = "lcp_trigrams.sav"
        self.cmudict = cmudict.dict()
        self.wnlp = WonderlicNLP()
        self.embeddings_index = {}
        self.wiki_top10 = [word[0].split()[0] for word in pd.read_csv("LCP/wiki_top10.csv").values]
        self.model = RandomForestRegressor(n_estimators=100)

    # GloVe setup

    def initialize_glove(self):
        print("INITIALIZING GLOVE...", datetime.now().strftime("%H:%M:%S"))
        f = open('LCP/glove.42B.300d.txt', encoding="utf8")
        for line in f:
            values = line.split()
            word = values[0]
            coefs = np.asarray(values[1:], dtype='float32') 
            self.embeddings_index[word] = coefs
        f.close()
        print("GLOVE READY!", datetime.now().strftime("%H:%M:%S"))

    def glove_embedding(self, word):
        embedding = [emb for emb in self.embeddings_index[str(word).lower()]] if str(word).lower() in self.embeddings_index.keys() else [-1 for i in range(300)]
        return embedding
    
    def find_word_pos(self, word, tokens):
        lemmatizer = WordNetLemmatizer()
        search_tokens = [lemmatizer.lemmatize(word) for word in tokens]
        if word in tokens:
            return tokens.index(word)
        elif word in search_tokens:
            return search_tokens.index(word)
        else:
            return None

    def extract_features(self, data):
        features = defaultdict(list)
        for id in tqdm(data.index, desc="PROCESSING DATA"):
            raw_token = "null" if str(data.loc[id]["token"]) == "nan" else str(data.loc[id]["token"])
            token = raw_token.lower()
            sent = data.loc[id]["sentence"]
            mrc_features = self.wnlp.get_mrc_features(token)
            glove = self.glove_embedding(token)

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
            features["wiki_fred"].append(int(token in self.wiki_top10))

            tokens = nltk.word_tokenize(re.sub(r"[^\w\s]", r" ", sent))
            word_pos = self.find_word_pos(raw_token, tokens)

            prev_word = tokens[word_pos-1] if word_pos!=0 and word_pos is not None else None
            next_word = tokens[word_pos+1] if word_pos!=(len(tokens)-1) and word_pos is not None else None

            prev_glove = self.glove_embedding(prev_word)
            for i in range(1,301):
                features[f"prev_glove{i}"].append(prev_glove[i-1])
            next_glove = self.glove_embedding(next_word)
            for i in range(1,301):
                features[f"next_glove{i}"].append(next_glove[i-1])

        return features

    def fit(self, train_data, train_labels):
        print("TRAINING...", datetime.now().strftime("%H:%M:%S"))
        self.initialize_glove()
        features = self.extract_features(train_data)
        self.model.fit(pd.DataFrame(features), train_labels)
        print("TRAINING DONE!", datetime.now().strftime("%H:%M:%S"))

    def to_likert(self, prediction):
        if prediction >= 0 and prediction < 0.2:
            return 1
        elif prediction >= 0.2 and prediction < 0.4:
            return 2
        elif prediction >= 0.4 and prediction < 0.6:
            return 3
        elif prediction >= 0.6 and prediction < 0.8:
            return 4
        else:
            return 5

    def predict(self, test_data, development=False):
        print("LOOKING INTO THE ORB...", datetime.now().strftime("%H:%M:%S"))
        tokens = test_data["token"]
        predictions = self.model.predict(pd.DataFrame(self.extract_features(test_data)))
        if not development:
            for i in range(len(predictions)):
                print(f"{tokens[i]} is a {self.to_likert(predictions[i])} on the Likert scale.")
        return predictions

    def score(self, train_data, train_labels):
        print("SCORING MODEL...", datetime.now().strftime("%H:%M:%S"))
        return self.model.score(pd.DataFrame(self.extract_features(train_data)), train_labels)
    
    def metrics(self, test_data, test_labels):
        labels_pred = self.predict(test_data, True)
        mae = mean_absolute_error(test_labels, labels_pred)
        rmse = math.sqrt(mean_squared_error(test_labels, labels_pred))
        print("MAE:", mae)
        print("RMSE:", rmse)
    
    def save(self):
        pickle.dump([self.model, self.embeddings_index], open(self.filename, "wb"))
    
    def load(self):
       data = pickle.load(open(self.filename, "rb"))
       self.model = data[0]
       self.embeddings_index = data[1]

    def feature_importances(self):
        return self.model.feature_importances_


def main():
    lcp = LCP()
    #lcp.load()
    train_data = pd.read_csv("LCP/CompLex/train/lcp_single_train.tsv", sep="\t", index_col="id", error_bad_lines=True, quoting=csv.QUOTE_NONE)
    train_labels = train_data["complexity"]
    lcp.fit(train_data, train_labels)
    print("R^2:", lcp.score(train_data, train_labels))
    test_data = pd.read_csv("LCP/CompLex/test-labels/lcp_single_test.tsv", sep="\t", index_col="id", error_bad_lines=True, quoting=csv.QUOTE_NONE)
    test_labels = pd.read_csv("LCP/CompLex/test-labels/lcp_single_test.tsv", sep="\t", error_bad_lines=True, quoting=csv.QUOTE_NONE)["complexity"]
    lcp.metrics(test_data, test_labels)
    #lcp.save()
    trial_data = pd.read_csv("LCP/CompLex/trial/lcp_single_trial.tsv", sep="\t", index_col="id", error_bad_lines=True, quoting=csv.QUOTE_NONE)
    #test_predict = [["ingenuous languid magnanimous nascent conflagration indefatigable", "indefatigable"]]
    #test_predict = pd.DataFrame(test_predict, columns=["sentence", "token"])

    print(lcp.predict(trial_data[0:5]))

    importance = lcp.feature_importances()
    glove_word = 0
    hc = 0
    glove_prev = 0
    glove_next = 0
    for i,v in enumerate(importance):
        if i <= 300:
            glove_word += v
        elif i > 300 and i <= 311:
            hc += v
        elif i > 311 and i <= 612:
            glove_prev += v
        else:
            glove_next += v
    print(glove_word, hc, glove_prev, glove_next)
    #plt.bar(["word_glove", "hc", "prev_glove", "next_glove"], [glove_word, hc, glove_prev, glove_next])
    #plt.show()



main()
