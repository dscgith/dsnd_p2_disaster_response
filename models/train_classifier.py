# import libraries
import sys
import pandas as pd
import numpy as np
import pickle
import re
from sqlalchemy import create_engine
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.tree import DecisionTreeClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.metrics import classification_report
from nltk.tokenize import word_tokenize
from nltk.stem.wordnet import WordNetLemmatizer


def load_data(database_filepath):
    """Load data from the database created by the ETL pipeline"""
    
    engine = create_engine('sqlite:///' + database_filepath)
    df = pd.read_sql_table('msg_cat', con=engine)
    
    X = df["message"]
    Y = df.iloc[:, 4:]
    
    return X, Y, Y.columns

    
def tokenize(text):
    """Tokenize given text"""
    
    # case normalization
    text = text.lower()
    # removing whitespace / punctuation
    whitespace = re.compile(r"[^\w]")
    text = re.sub(whitespace, " ", text)
    # tokenize text
    text = word_tokenize(text)
    # stopword removal takes place in CountVectorizer
    # lemmatize words
    text = [WordNetLemmatizer().lemmatize(w) for w in text]
        
    return text


def build_model():
    """Return model based on best classifier pipeline found"""
    
    # NLP pipeline parameters have been optimized in notebook
    pipeline_v3 = Pipeline([
        ("features", FeatureUnion([
            ("NLP", Pipeline([
                ("vec", CountVectorizer(ngram_range=(1,2), 
                                        stop_words='english',
                                        tokenizer=tokenize)), 
                ("tfidf", TfidfTransformer(smooth_idf=False, 
                                           use_idf=False))
            ])),
        ])),
        ("clf", DecisionTreeClassifier(random_state=42))
    ])
    # classifier parameters to tweak by GridSearchCV
    parameters_v3 = {
        "clf__min_samples_split": [2, 3]
    }
    
    cv = GridSearchCV(estimator=pipeline_v3, param_grid=parameters_v3)
    
    return cv


def evaluate_model(model, X_test, Y_test, category_names):
    """Evaluate model on test data"""
    
    ypred = model.predict(X_test)
    for i in range(ypred.shape[1]):
        print("### " + category_names[i] + " ###\n" + \
              classification_report(Y_test.values[:, i], ypred[:, i]) + "\n")    
        

def save_model(model, model_filepath):
    """"Save trained model to pickle dump"""

    pickle.dump(model, open(model_filepath, "wb"))
              

def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

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