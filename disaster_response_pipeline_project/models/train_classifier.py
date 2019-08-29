import sys
# import libraries
import pandas as pd
from sqlalchemy import create_engine
from nltk.tokenize import word_tokenize
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.svm import LinearSVC
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.multioutput import MultiOutputClassifier
from sklearn.model_selection import train_test_split
from sklearn import metrics
from nltk.stem import WordNetLemmatizer
from sklearn.metrics import precision_recall_fscore_support as score
from sklearn.model_selection import GridSearchCV
from sklearn.externals import joblib
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
import nltk
nltk.download('punkt')
nltk.download('wordnet')



def load_data(database_filepath):
    '''
    Inputs:
        database_filepath: the file path to the database 
    Outputs: 
         X: data used for training
         Y: labels for data
         c: list of classes'''
    engine = create_engine(database_filepath)
    df = pd.read_sql_table('msg',engine)
    X = df['message']
    Y = df.drop(['id', 'message', 'original', 'genre'], axis = 1)
    c = Y.columns.values
    return X,Y,c


def tokenize(text):
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()
    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)
    return clean_tokens

def build_model():
    pipeline = Pipeline([
    ('vectorizer', CountVectorizer(tokenize)),
    ('tfidf', TfidfTransformer()),
    ('clf', MultiOutputClassifier(OneVsRestClassifier(LinearSVC())))])
    parameters = {
    'clf__estimator__estimator__class_weight': ['balanced', None],
    'clf__estimator__estimator__C': [0.01, 0.1, 1],
    'clf__estimator__estimator__max_iter': [50, 80],
    'clf__estimator__estimator__tol': [1e-4],
    }
    cv = GridSearchCV(pipeline, param_grid=parameters)
    return cv


def evaluate_model(model, X_test, Y_test, category_names):
    '''
    Inputs:
        model: the trained model
        X_test:  test data
        Y_test: the actual labels  
        claesses: class names
    Outputs: a df containing F1 score, percison, recall for each class '''
    Y_prep = model.predict(X_test)
    report = []
    for i in range(len(category_names)):
        precision,recall,fscore,support=score(Y_test[category_names[i]],Y_prep[:,i],average='macro')
        report.append([category_names[i],precision,recall,fscore])
    results = pd.DataFrame(report, columns =['Class','Precision','Recall','F score']).set_index('Class')
    return results

def save_model(model, model_filepath):
    '''
    Inputs:
        model: the trained model
        model_filepath:  file path for the model to be saved to
    Outputs: pickled file '''
    joblib.dump(model, model_filepath)
    return


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
