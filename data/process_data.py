import sys
import pandas as pd
import numpy as np
from sqlalchemy import create_engine


def load_data(messages_filepath, categories_filepath):
    """Load messages and categories data and return merged DataFrame"""
    
    # read csv files
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    # create merged DataFrame
    df = messages.merge(categories, on=["id"])
    
    return df


def clean_data(df):
    """Create a clean DataFrame without duplicates"""
    
    # reorganize category values to separate columns wit 0 / 1 values
    categories = df.categories.str.split(";", expand=True)
    row = categories.iloc[0]
    category_colnames = row.apply(lambda x: x.split("-")[0])
    categories.columns = category_colnames
    categories = categories.applymap(lambda x: int(x.split("-")[1]))
    
    # integrate clean categories into DataFrame
    df.drop(columns=["categories"], inplace=True)
    df = pd.concat([df, categories], axis=1, sort=False)
   
    # remove duplicate values
    duplicate_bool = df.duplicated()
    df = df[~duplicate_bool]
    
    # drop rows where feature "related" == 2, which doesn't make sense
    df = df[~(df.related == 2)]
    # drop "child_alone" feature, since all values are 0 => no classification possible
    df.drop(columns=["child_alone"], inplace=True)
    
    return df


def save_data(df, database_filename):
    """Save DataFrame to SQL database"""
   
    engine = create_engine('sqlite:///' + database_filename)
    df.to_sql('msg_cat', engine, index=False)  


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