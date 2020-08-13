import sys
import pandas as pd
import numpy as np

from sqlalchemy import create_engine

def load_data(messages_filepath, categories_filepath):
    """
        Load data from the csv. 
    Args: 
        messages_filepath: the path of the messages.csv file
        categories_filepath: the path of the categories.csv file
    Returns: 
        df (DataFrame): dataframe that results in merging of messages and categories
    """
    
    # load the messages and categories csv files
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    
    # Now merge the two together
    df = messages.merge(categories, on = 'id')
    
    return df


def clean_data(df):
    """
        Clean the merged dataframe 
    Args: 
        df: Uncleaned merged dataframe
    Returns: 
        df (Dataframe): The cleaned merged dataframe
    """
    
    # Split categories into multiple columns (36)
    categories = df['categories'].str.split(';', expand = True)
    
    # Get new column names
    row = categories.loc[0,:]
    category_colnames = row.apply(lambda x: x.split('-')[0])
    categories.columns = category_colnames
    
    # Convert category values to just 0 or 1
    for column in categories:
        # use the last character in the string (That is where 1 or 0 is located)
        categories[column] = categories[column].apply(lambda x: int(x.split('-')[1]))

        
    # Answer from mentor: There are some related columns that have the value 2, that shouldn't be the case
    # So change the 2 to 1 in all instances
    categories['related'] = categories['related'].replace(2, 1)

    # Replace the categories column with the new 36 unique categories
    
    # drop the original categories column from `df`
    df.drop(columns=['categories'], inplace=True)
    
    # concatenate the original dataframe with the new `categories` dataframe
    df = pd.concat([df, categories], axis=1)
    
    # drop duplicates
    df.drop_duplicates(inplace=True)
    
    return df

    
def save_data(df, database_filename):
    """
        Save the cleaned dataframe into sqlite database
    Args: 
        df: The cleaned merged dataframe
        database_filename: the database name
    Returns: 
        None
    """
    engine = create_engine('sqlite:///{}'.format(database_filename))
    df.to_sql('messages_cleaned_relate', engine, index=False)  


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