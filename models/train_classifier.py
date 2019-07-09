import sys
import pandas as pd
from sqlalchemy import create_engine

import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.stem.wordnet import WordNetLemmatizer

from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report
from sklearn.base import BaseEstimator, TransformerMixin

import pickle



def load_data(database_filepath):
    engine = create_engine('sqlite:///' + database_filepath)
    try:
        df = pd.read_sql_table('SELECT * FROM DisasterResponse', engine)
    except:
        df = pd.read_csv("models/DisasterResponse.csv", dtype = {'message': str, 'original': str, 'genre': str})
    
    X = df["message"]
    y = df.iloc[:, 4:]
    category_names = y.columns
    
    return X, y, category_names



def tokenize(text):
    """
    Tokenize text
    
    Input: raw text
    Output: Normalized, and Lemmatized text
    """
    # Normalize text
    text = re.sub(r"[^a-zA-Z0-9]", " ", text.lower())
    # Tokenize text
    words = text.split()
    # Remove stop words
    words = [w for w in words if w not in stopwords.words("english")]
    # Reduce words to their root form
    lemmed = [WordNetLemmatizer().lemmatize(w) for w in words]
    # Lemmatize verbs by specifying pos
    lemmed = [WordNetLemmatizer().lemmatize(w, pos='v') for w in lemmed]
    
    return lemmed


class StartingVerbExtractor(BaseEstimator, TransformerMixin):
    """
    Customized transformer for extracting starting verbs
    """
    def starting_verb(self, text):
        sentence_list = nltk.sent_tokenize(text)
        for sentence in sentence_list:
            pos_tags = nltk.pos_tag(tokenize(sentence))
            try:
                first_word, first_tag = pos_tags[0]
                if first_tag in ['VB', 'VBP'] or first_word == 'RT':
                    return True
            except:
                continue
        return False
    
    def fit(self, x, y=None):
        return self
    
    def transform(self, X):
        X_tagged = pd.Series(X).apply(self.starting_verb)
        return pd.DataFrame(X_tagged)
    
        
def build_model():
    """
    1. Build pipeline to take in the message column as input and 
       output classification results on the other 36 categories 
       in the dataset.
    2. Use grid search to find better parameters.
    """
    nltk.download(['punkt', 'wordnet', 'averaged_perceptron_tagger', 'stopwords'])
    pipeline = Pipeline([
        ('features', FeatureUnion([
        
            ('text_pipeline', Pipeline([
                ('vect', CountVectorizer()),
                ('tfidf', TfidfTransformer())
            ])),
            
            ('strarting_verb', StartingVerbExtractor())
            
        ])),

         ('clf', MultiOutputClassifier(estimator = AdaBoostClassifier(random_state = 42)))

    ])
    
    parameters = {"clf__estimator__learning_rate": [0.1, 0.5, 1.0],
                  "clf__estimator__n_estimators": [25, 50, 75]
                  }
    
    from sklearn.model_selection import GridSearchCV
    cv = GridSearchCV(pipeline, param_grid = parameters)    
    
    return cv



def evaluate_model(model, X_test, Y_test, category_names):
    """
    Evalueate model by running on X_test and Y_text, 
    and metrics used for check are f1 score, precision and recall

    Inputs:
    - model, X_test, Y_text, category_names

    Outputs:
    - f1 score, precision and recall of input model
    """
    y_pred = pd.DataFrame(data = model.predict(X_test), columns = category_names)

    precision, recall, f1_score = [], [], []

    for category in category_names:
        scores = classification_report(Y_test[category], y_pred[category])
        precision.append([x for x in scores.strip().split("avg / total")[1].strip().split(" ") 
                          if len(x) > 0][:3][0])
        recall.append([x for x in scores.strip().split("avg / total")[1].strip().split(" ") 
                       if len(x) > 0][:3][1])
        
    model_metric = pd.concat([
        pd.DataFrame(data = [precision, recall], index = ["precision", "recall"], 
                     columns = category_names),
        (Y_test.reset_index() == y_pred.reset_index()).mean()[1:].to_frame("accuracy").T
    ])

    for col in model_metric.columns:
        model_metric[col] = model_metric[col].astype(float)

    return model_metric    



def save_model(model, model_filepath):
    """
    Save model to desitination
    """
    pickle.dump(model, open(model_filepath, 'wb'))


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