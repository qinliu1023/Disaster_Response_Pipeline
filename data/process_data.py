import sys
import pandas as pd
import numpy as np
from sqlalchemy import create_engine

import os

def load_data(messages_filepath, categories_filepath):
    """
    1. Read in raw csv files
    2. Drop duplcates in Each Dateframe

    Inputs:
    - filepath of messages, categories

    Output:
    - dataframe messages, categories
    """
    messages = pd.read_csv(messages_filepath).drop_duplicates(keep = "first")
    categories = pd.read_csv(categories_filepath).drop_duplicates(keep = "first")

    df = messages.merge(categories, how = "inner", on = "id")
    
    return df


def clean_data(df):
    """
    Transform category information into 36 separate columns
    """  
    # get category_colnames by 
    # 1. taking one value from categories column
    # 2. split it into 36 values
    # 3. taking the first part (prior to '-') as category colnames
    category_colnames = df.head(1)["categories"].str.split(";", expand = True)\
    .apply(lambda x: x.str.split("-", expand = True)[0], axis = 0).values[0]
    # split categories column into 36 columns and assign it to new columns in category_colnames
    df[category_colnames] = df["categories"].str.split(";", expand = True)\
    .apply(lambda x: x.str.split("-", expand = True)[1]).astype(int)
    # drop original categories column from table categories
    df.drop("categories", axis = 1, inplace = True)

    return df


def save_data(df, database_filename):
    """
    Save dataframe to database
    """  
    engine = create_engine('sqlite:///'+database_filename)
    df.to_sql(database_filename, engine, index=False, if_exists = "replace" )
    df.to_csv('models/DisasterResponse.csv', index=False)

def main():
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)
        
        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)
        
        print('Cleaned data saved to database!')
    
    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')


if __name__ == '__main__':
    main()