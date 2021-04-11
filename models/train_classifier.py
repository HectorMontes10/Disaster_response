import sys
import subprocess
import sklearn
from sqlalchemy import create_engine
import pandas as pd
import numpy as np
import nltk
import re
from nltk.tokenize import word_tokenize
from nltk.tokenize import sent_tokenize
from nltk import pos_tag
from nltk.stem.wordnet import WordNetLemmatizer
from sklearn.datasets import make_multilabel_classification
from sklearn.multioutput import MultiOutputClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.model_selection import train_test_split
from time import time 
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.metrics import classification_report
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import GridSearchCV
import pickle
nltk.download(['punkt', 'wordnet'])
nltk.download('averaged_perceptron_tagger')

def load_data(database_filepath):
    engine = create_engine('sqlite:///'+database_filepath)
    df = pd.read_sql_table("Cleaned_messages",con=engine)

    X = df.message.values
    Y = df.iloc[:,4:]
    category_names = Y.columns
    
    return X, Y, category_names

def tokenize(text):
    # get list of all urls using regex
    url_regex = 'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
    detected_urls = re.compile(url_regex).findall(text)
    # replace each url in text string with placeholder
    for url in detected_urls:
        text = text.replace(url,"urlplaceholder")
    # tokenize text
    tokens = word_tokenize(text)
    # initiate lemmatizer
    lemmatizer = WordNetLemmatizer()
    # iterate through each token
    clean_tokens = []
    for tok in tokens:
        # lemmatize, normalize case, and remove leading/trailing white space
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)
    return clean_tokens

class DummyEstimator(BaseEstimator):
    def fit(self): pass
    def score(self): pass

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

class CalculateLengthText(BaseEstimator, TransformerMixin):
    
    def __init__(self):
        
        self.min_length =  0
        self.max_lenth = 0
        
    def length_text(self, text):
        return len(text)

    def fit(self, X, y=None):
        Length_X = [self.length_text(text) for text in X]
        self.min_length = min(Length_X)
        self.max_length = max(Length_X)
        return self

    def transform(self, X):
        # apply starting_verb function to all values in X
        Length_X = [self.length_text(text) for text in X]
        Length_X_scaled = [(length-self.min_length)/(self.max_length-self.min_length) for length in Length_X]
        return pd.DataFrame(Length_X_scaled)

def build_model():
    
    pipeline = Pipeline([
        ('features', FeatureUnion([

            ('text_pipeline', Pipeline([
                ('vect', CountVectorizer(tokenizer=tokenize)),
                ('tfidf', TfidfTransformer())
            ])),
            ('starting_verb', StartingVerbExtractor()),
            ('length_text', CalculateLengthText())
        ])),
        ('clf', DummyEstimator())
    ])

    n_estimators = [100,200,300]
    max_depth =[10,20]
    max_iter = [200,300]
    class_weight = [None, 'balanced']

    # Candidate learning algorithms and their hyperparameters

    search_space = [{'clf': [MultiOutputClassifier(RandomForestClassifier())],
                    'clf__estimator__n_estimators': n_estimators,
                    'clf__estimator__max_depth': max_depth},
                    {'clf': [MultiOutputClassifier(LogisticRegression())],
                     'clf__estimator__max_iter': max_iter,
                     'clf__estimator__class_weight': class_weight}]

    #Create grid search
    cv = GridSearchCV(pipeline, search_space, n_jobs=-1)
    return cv

def evaluate_model(model, X_test, Y_test, category_names):
    best_pipeline = model.best_estimator_
    Y_pred = best_pipeline.predict(X_test)
    target_names = category_names
    result = classification_report(Y_test,Y_pred,target_names = target_names, zero_division = 0)
    return result

def save_model(model, model_filepath):
    pickle.dump(model, open(model_filepath, 'wb'))

def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        #print(X)
        #print(Y)
        #print(category_names)
        
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        start_time = time()
        model.fit(X_train, Y_train)
        end_time = time()
        print("The time for training was: {}".format(end_time-start_time))
        
        print('Evaluating model...')
        print(evaluate_model(model, X_test, Y_test, category_names))

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')

if __name__ == '__main__':
    main()