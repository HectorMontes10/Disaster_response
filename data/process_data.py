import sys
import pandas as pd
import numpy as np
import json
from sqlalchemy import create_engine

def load_data(messages_filepath, categories_filepath):
    
    '''
    
    This function load the datasets messages and categories and merge based on id column.
    
    Params:
        messages_filepath (str): String that contain the path to messages file
        categories_filepath (str): String that contain the path to categories file
    Returns:
        df (pandas DataFrame): DataFrame with columns: id,message,original,genre,categories
                               row: A single messages
                               columns:
                                    id-->Id for each message
                                    messages--> Text of message
                                    categories --> A single column containing the categories marks for the message
                                    
    '''
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    df = messages.merge(categories, on="id")
    return df
    
def clean_data(df):
    
    '''
    This function clean the files messages and categories. The cleaning operations are:
        
        1. Split the single categorie column in multiple categories columns
        2. Rename the categories columns using the first row
        3. Convert the categories columns in ints variables.
        4. Drop the single original categorie column and replace it with multiples categories columns
        5. Remove duplicates rows.
        6. Remove constant columns.
        7. Set "related" categorie to binary (it has some values that are not 0,1)
    
    Params:
        df (pandas DataFrame): DataFrame over the cleaning operations are made.
    Returns
        df (pandas DataFrame): DataFrame cleaned.
        
    '''
    # create a dataframe of the 36 individual category 
    categories =  df.categories.str.split(';', expand=True)
    #Rename the categories columns
    row = categories.iloc[0,]
    category_colnames = row.apply(lambda x: x[:-2])
    categories.columns = category_colnames
    #Convert columns to ints
    for column in categories:
        # set each value to be the last character of the string
        categories[column] = categories[column].apply(lambda x: x[-1])
        # convert column from string to numeric
        categories[column] = categories[column].astype(int)
    
    #Remove original categorie column
    df = df.drop(columns=['categories'])
    #Insert the categories columns created
    df = pd.concat([df,categories],axis=1)
    #Remove duplicated rows
    df = df.drop_duplicates()
    #Remove constants columns
    col_to_drop = []
    for col in categories.columns:
        if len(categories[col].unique())==1:
            col_to_drop.append(col)
    df = df.drop(columns=col_to_drop)
    #Set "related" categorie to binary
    df['related']=df['related'].map(lambda x: 1 if x==2 else x)
    return df

def save_data(df, database_filename):
    
    '''
    This function save a dataframe object like a table of SQLite database
    
    Params:
        df (pandas DataFrame): A pandas DataFrame for to save in table of SQLite database
        database_filename (str): A string that contain the relative path where we will be save the table
    Returns:
        Without returns, this is a procedure.
    
    '''
    engine = create_engine('sqlite:///'+database_filename)
    df.to_sql('Cleaned_messages', engine, index=False)

def main():
    
    '''
    
    This function control the ETL flow and call the other functions for load, clean, and save data
    
    '''
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