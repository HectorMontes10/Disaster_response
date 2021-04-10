from sklearn.base import BaseEstimator, TransformerMixin
import pandas as pd

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