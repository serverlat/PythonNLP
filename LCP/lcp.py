import pandas as pd
import pickle
import numpy as np
import cmudict
import torch 
import math
from datetime import datetime
from models import InferSent
from sklearn.preprocessing import StandardScaler
from wonderlic_nlp import WonderlicNLP 
from collections import defaultdict
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_absolute_error, mean_squared_error
from tqdm import tqdm

class LCP:

    def __init__(self):
        self.cmudict = cmudict.dict()
        self.wnlp = WonderlicNLP()
        self.embeddings_index = {}
        self.wiki_top10 = [word[0].split()[0] for word in pd.read_csv("LCP/wiki_top10.csv").values]
        self.infersent_model_path = 'LCP/infersent%s.pkl' % 1
        self.infersent_model_params = {'bsize': 64, 'word_emb_dim': 300, 'enc_lstm_dim': 2048,
                                        'pool_type': 'max', 'dpout_model': 0.0, 'version': 1}
        self.infersent = InferSent(self.infersent_model_params)
        self.model = RandomForestRegressor(n_estimators=16)

    #InferSent setup 

    def initialize_infersent(self, sentences):
        print("INITIALIZING INFERSENT...", datetime.now().strftime("%H:%M:%S"))
        self.infersent.load_state_dict(torch.load(self.infersent_model_path))
        w2v_path = 'LCP/glove.42B.300d.txt'
        self.infersent.set_w2v_path(w2v_path)
        self.infersent.build_vocab(sentences, tokenize=True) 
        print("INFERSENT READY!", datetime.now().strftime("%H:%M:%S"))

    def infersent_embedding(self, sentence):
        return self.infersent.encode(sentence, tokenize=True) # one sentence? 4096 features, according to paper

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
        return [emb for emb in self.embeddings_index[str(word).lower()]]

    def extract_features(self, data):
        features = defaultdict(list)
        for id in tqdm(data.index, desc="PROCESSING DATA"):
            token = str(data.loc[id]["token"]).lower()
            sent = data.loc[id]["sentence"]
            mrc_features = self.wnlp.get_mrc_features(token)
            glove = self.glove_embedding(token)
            #infersent = self.infersent_embedding([sent])[0]
            #for i in range(1, 4097): #4096 dim
                #features[f"infersent{i}"].append(infersent[i-1])
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
        return features

    # Actual model
    def fit(self, train_data, train_labels):
        print("TRAINING...", datetime.now().strftime("%H:%M:%S"))
        self.initialize_glove()
        #self.initialize_infersent(train_data["sentence"])
        self.model.fit(pd.DataFrame(self.extract_features(train_data)), train_labels)
        print("TRAINING DONE!", datetime.now().strftime("%H:%M:%S"))

    def predict(self, test_data):
        print("LOOKING INTO THE ORB...", datetime.now().strftime("%H:%M:%S"))
        return self.model.predict(pd.DataFrame(self.extract_features(test_data)))

    def score(self, train_data, train_labels):
        print("SCORING MODEL...", datetime.now().strftime("%H:%M:%S"))
        return self.model.score(pd.DataFrame(self.extract_features(train_data)), train_labels)
    
    def metrics(self, test_data, test_labels):
        labels_pred = self.predict(test_data)
        mae = mean_absolute_error(test_labels, labels_pred)
        rmse = math.sqrt(mean_squared_error(test_labels, labels_pred))
        print("MAE:", mae)
        print("RMSE:", rmse)

def main():
    lcp = LCP()
    train_data = pd.read_csv("LCP/CompLex/train/lcp_single_train.tsv", sep="\t", index_col="id")
    train_labels = train_data["complexity"]
    lcp.fit(train_data, train_labels)
    print("R^2:", lcp.score(train_data, train_labels))
    test_data = pd.read_csv("LCP/CompLex/test-labels/lcp_single_test.tsv", sep="\t", index_col="id")
    test_labels = pd.read_csv("LCP/CompLex/test-labels/lcp_single_test.tsv", sep="\t")["complexity"]
    lcp.predict(test_data)
    lcp.metrics(test_data, test_labels)

main()
