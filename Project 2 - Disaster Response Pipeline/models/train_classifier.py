import pandas as pd
import numpy as np
import pickle
import nltk
import sys

from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from sklearn.pipeline import Pipeline
from sqlalchemy import create_engine

from tqdm import tqdm

def load_data(database_filepath):
    """
        Loading the data from sqlite database that was created in process_data.py
    Args: 
        database_filepath: path of the file for the database
    Returns: 
        X (DataFrame): messages  in database
        Y (DataFrame): categories in database
        category_names (List): List of all the category names
    """
    engine = create_engine('sqlite:///{}'.format(database_filepath))
    df = pd.read_sql_table('messages_cleaned_relate', engine)
    X = df['message']
    Y = df.drop(columns=['id', 'message', 'original', 'genre'])
    category_names = Y.columns
    
    return X, Y, category_names

def tokenize(text):
    """
        Tokenizing the text
    Args: 
        text: text to be tokenized
    Returns: 
        cleaned_text (array): The array of cleaned tokens
    """
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()
    
    # Cleaning the text
    cleaned_text = []
    for token in tokens:
        # Lowercasing and removing leading/trailing whitespaces
        clean_token = lemmatizer.lemmatize(token).lower().strip()
        # Appending to the array of cleaned tokens
        cleaned_text.append(clean_token)
                   
    return cleaned_text

def build_model():
    """
      Building our machine learning pipeline
    Args: 
        None
    Returns: 
        model: classifier object
    """   
    # Creating a Pipeline inspired from the lessons leading up to the project
    # Resource used for RandomForestClassifier: https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html
    pipeline = Pipeline([
        ('text_pipeline', Pipeline([
            ('vect', CountVectorizer(tokenizer=tokenize)),
            ('tfidf', TfidfTransformer())
        ])),

        ('clf', MultiOutputClassifier(RandomForestClassifier(n_estimators = 100)))
    ]) 

    parameters = {
        'clf__estimator__n_estimators':[50, 100],
        'text_pipeline__tfidf__use_idf': (True, False)
    }
    
    # Creating the model (also verbose = 10 is there to just give some insight into how much is left in training)
    model = GridSearchCV(pipeline, param_grid = parameters, cv = 2, n_jobs = 5, verbose = 10)

#     return model

    return model

def evaluate_model(model, X_test, Y_test, category_names):
    """
        Evaluate the model performances and display the classification report for the model
    Args: 
        model: model being evaluated
        X_test: X_test df
        Y_test: Y_test df
        category_names: list of category names
    Returns: 
        None
    """   
    
    # Predict
    Y_predicted = model.predict(X_test)
    
    # Printing the classification report
    # Note: Classification report was imported from 
    print(classification_report(Y_test, Y_predicted, target_names=category_names))

def save_model(model, model_filepath):
    """
        Save the model to a pickle file
    Args: 
        model: model being saved
        model_filepath: filepath for the saved model
    Returns: 
        None
    """   
    
    # The filename for the model pkl file
    model_pkl = str(model_filepath)
    
    # Dumping the model into a pickle file
    with open(model_pkl, 'wb') as model_file:
        pickle.dump(model, model_file)

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
