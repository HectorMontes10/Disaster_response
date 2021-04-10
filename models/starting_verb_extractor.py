from sklearn.base import BaseEstimator, TransformerMixin
from nltk.tokenize import sent_tokenize
from nltk import pos_tag
import pandas as pd
from nltk.tokenize import word_tokenize

class StartingVerbExtractor(BaseEstimator, TransformerMixin):

    def starting_verb(self, text):
        # tokenize by sentences
        sentence_list = sent_tokenize(text)
        for sentence in sentence_list:
            # tokenize each sentence into words and tag part of speech
            pos_tags = pos_tag(word_tokenize(sentence))

            # index pos_tags to get the first word and part of speech tag
            first_word, first_tag = pos_tags[0]
            
            # return true if the first word is an appropriate verb or RT for retweet
            if first_tag in ['VB', 'VBP'] or first_word == 'RT':
                return 1
        return 0

    def fit(self, x, y=None):
        return self

    def transform(self, X):
        # apply starting_verb function to all values in X
        X_tagged = [self.starting_verb(text) for text in X]

        return pd.DataFrame(X_tagged)