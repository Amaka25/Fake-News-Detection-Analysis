import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

class NumericFeatures(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self
    
    def transform(self, X, y=None):
        df = pd.DataFrame(X, columns=["title"])
        
        df["title_length"] = df["title"].apply(lambda x: len(str(x)))
        df["word_count"] = df["title"].apply(lambda x: len(str(x).split()))
        df["exclamation_count"] = df["title"].str.count("!")
        df["question_count"] = df["title"].str.count("\?")
        
        return df[["title_length", "word_count", "exclamation_count", "question_count"]].values
