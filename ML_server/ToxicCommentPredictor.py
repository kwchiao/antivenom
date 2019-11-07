import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt

import re
from nltk.stem import WordNetLemmatizer 
import nltk
nltk.download('wordnet') 

from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import csr_matrix, hstack
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression


import sys, os, re, csv, codecs, numpy as np, pandas as pd
#import matplotlib.pyplot as plt
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Dense, Input, LSTM, Embedding, Dropout, Activation, GRU
from keras.layers import Bidirectional, GlobalMaxPool1D, Conv1D, MaxPooling1D
from keras.models import Model, Sequential, load_model
from keras.callbacks import EarlyStopping

from keras import initializers, regularizers, constraints, optimizers, layers

from sklearn.metrics import roc_auc_score




import pickle

def saveobj(save_list, filename):
    with open(filename, 'wb') as f:  # Python 3: open(..., 'wb')
        pickle.dump(save_list, f)
        
# restore object        
def loadobj(filename):
    with open(filename, 'rb') as f:  # Python 3: open(..., 'rb')
        li = pickle.load(f)
    return li


class ToxicCommentPredictor:
    def __init__(self):
        self.classes = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']
        # load preprocessed pickle
        self.train_df, self.valid_df, self.test_df = loadobj('data/filtered_comment_pickle')
        # RNN initit
        # self.Y_tr = self.train_df[self.classes].values
        # self.Y_v = valid_df[classes].values
        # self.Y_te = test_df[classes].values
        self.list_sentences_train = self.train_df["filt_comment"]
        # self.list_sentences_valid = valid_df["filt_comment"]
        # self.list_sentences_test = test_df["filt_comment"]

        self.batch_size = 32

        self.max_features = 20000
        self.rnn_tokenizer = Tokenizer(num_words=self.max_features)
        self.rnn_tokenizer.fit_on_texts(list(self.list_sentences_train))
        self.rnn_list_tokenized_train = self.rnn_tokenizer.texts_to_sequences(self.list_sentences_train)
        # self.rnn_list_tokenized_valid = self.rnn_tokenizer.texts_to_sequences(self.list_sentences_valid)
        # self.rnn_list_tokenized_test = self.rnn_tokenizer.texts_to_sequences(self.list_sentences_test)

        self.rnn_maxlen = 200
        # self.rnn_X_tr = pad_sequences(rnn_list_tokenized_train, maxlen=rnn_maxlen)
        # self.rnn_X_v = pad_sequences(rnn_list_tokenized_valid, maxlen=rnn_maxlen)
        # self.rnn_X_te = pad_sequences(rnn_list_tokenized_test, maxlen=rnn_maxlen)

        self.gru_model = load_model('model/gru_model.h5')
        '''
        # cnn intialization
        self.cnn_tokenizer = Tokenizer(num_words=self.max_features,char_level=True)
        self.cnn_tokenizer.fit_on_texts(list(self.list_sentences_train))
        self.cnn_list_tokenized_train = self.cnn_tokenizer.texts_to_sequences(self.list_sentences_train)
        self.cnn_maxlen = 500
        self.cnn_model = load_model('model/cnn_model.h5')
        '''
        print('done')
        # logistic initialization

    def preprocessing(self, message):
        return message

    def rnn_prediction(self, message):
        list_tokenized = self.rnn_tokenizer.texts_to_sequences([message])
        x_cell = pad_sequences(list_tokenized, maxlen=self.rnn_maxlen)
        y_cell_predict = self.gru_model.predict(x_cell, batch_size=self.batch_size, verbose=0)
        return y_cell_predict[0]

    def cnn_prediction(self, message):
        list_tokenized = self.cnn_tokenizer.texts_to_sequences([message])
        x_cell = pad_sequences(list_tokenized, maxlen=self.cnn_maxlen)
        y_cell_predict = self.cnn_model.predict(x_cell, batch_size=self.batch_size, verbose=0)
        return y_cell_predict[0]
        
    def logistic_prediction(self, message):
        return 0
