#Import packages

from time import time
import re
from sqlalchemy import create_engine
import pandas as pd
import numpy as np
import sklearn
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV
import pickle

#Import custom class and functions

import sys
sys.path.append("./models")
from starting_verb_extractor import StartingVerbExtractor
from calculate_textlength import CalculateLengthText
from dummy_estimator  import DummyEstimator
import tokenize_
from tokenize_ import tokenize

def load_data(database_filepath):
    
    '''
    
    This function load a database of cleaned messages.
    
    Params:
        database_filepath (string): Path to sqlLite database
    Returns:
        X(numpy array): array with raw text to train model
        Y(numpy array): matrix with target variables (one per categorie)
        categorie_names(list): Names of target variables usefull for graphics
        
    '''
    engine = create_engine('sqlite:///'+database_filepath)
    df = pd.read_sql_table("Cleaned_messages",con=engine)

    X = df.message.values
    Y = df.iloc[:,4:]
    category_names = Y.columns
    
    return X, Y, category_names

def build_model():
    
    '''
    This function construct a pipeline with custom transformer and estimators. The pipeline is passed to a grid search function
    for tuning parameter for estimators. The pipeline include FeatureUnion based in custom transformer.
    
    Params:
        None
    Returns:
        cv(GridSearch object): An object of class GridSearch fitting over train data. The object have an attribute "best_estimator_"
                               that contain the best model finded.
    
    '''
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
    
    '''
    This function evaluate the best model in gridsearch object fitted over train data using "classification_report" function over
    X_test dataset.
    
    Params:
        model(gridSearch object): gridSearch object fitted over train data.
        X_test(numpy array): array of string to be used for test model
        Y_test(numpy array): Matrix with test dataset for evaluate model
        categorie_names(list): Names of target variables
    Return:
        result (string): Printed string with metrics for model evaluated (Recall, Precission and F1 Score) for each
                         target variable in model.
    
    '''
    best_pipeline = model.best_estimator_
    Y_pred = best_pipeline.predict(X_test)
    target_names = category_names
    result = classification_report(Y_test,Y_pred,target_names = target_names, zero_division = 0)
    return result

def save_model(model, model_filepath):
    
    '''
    
    This function save the gridSearch object. The object will be used to classify new messages passed by the user from the
    application's graphical interface.
    
    Params:
        model (gridSearch object): Contain the best fitted gridSearch object over train data
        model_filepath: Path where the object will be stored.
    Return:
        None: This is a procedure. it store a pickle object that contain the model in the model_filepath location.
    
    '''
    pickle.dump(model, open(model_filepath, 'wb'))

def main():
    
    '''
    
    This function control the training flow and call the other functions for load, train, and save model
    
    '''
    
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