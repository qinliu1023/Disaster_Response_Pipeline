# Disaster Response Pipeline Project


### Installation
Beside Python, nltk needs to be imported for downloading `punkt`, `wordnet`, `averaged_perceptron_tagger`, `stopwords` with
```python
import nltk
nltk.download(['punkt', 'wordnet', 'averaged_perceptron_tagger', 'stopwords'])
```

### Motivation
Build a machine learning pipeline to categorize emergency messages based on the needs communicated by the sender by utilizing new learned knowledge about ETL Pipeline, NLP Pipeline, ML Pipeline, and software engineer.


### File Description
1. data
- disaster_categories.csv: categories data csv file
- disaster_messages.csv: messages data csv file
- process_data.py: ETL pipeline takes in above two csv files as inputs and outputs cleaned and integrated data into a database
1. Models
- train_classifier.py: machine learning pipeline takes in the message column as input and outputs classification results on the other 36 categories in the dataset.
1. App
- run.py: file for running web application


### Instructions:
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. Go to http://0.0.0.0:3001/


### Licensing, Authors, Acknowledgements
Data used are provided by Udacity.
