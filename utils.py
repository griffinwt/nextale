
import pandas as pd
import numpy as np
import requests
from bs4 import BeautifulSoup, NavigableString, Tag
import re
import time
import random
import sys
pd.set_option('display.max_colwidth', None)

import re
from nltk.corpus import stopwords
swords = set(stopwords.words('english')) #set nltk stopword list equal to a variable

from nltk.stem import WordNetLemmatizer
from nltk.tokenize import RegexpTokenizer
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

lemmatizer = WordNetLemmatizer()
tokenizer = RegexpTokenizer(r'\w+') #create tokenizer to remove punctuation

def tokem_lite(some_string):
    stok = tokenizer.tokenize(some_string)
    #stok = ' '.join(stok) #return string of words
    slem = [lemmatizer.lemmatize(word) for word in stok]
    #slem = lemmatizer.lemmatize(stok)
    cleansw=[word for word in slem if word not in swords]
    return ' '.join(cleansw)
    for item in stok:
        slems = []
        for word in item:
            slems.append(lemmatizer.lemmatize(word)) #make list of non-stop words
            
    return ' '.join(slems) #return string of words

def clean_amazon_data(file_name, new_name):
    df = pd.read_csv(file_name, sep='\t', compression='gzip', error_bad_lines=False, low_memory=False) #read in file
    df.drop(columns=['marketplace', 'vine', 'product_category'], inplace=True) #drop columns that won't be used
    df['verified_purchase']=df['verified_purchase'].map({'Y':1, 'N':0}) #change verified_purchase to 1/0 classifier
    df['review_date'] = pd.to_datetime(df['review_date']) #convert review date to date time object
    print(f'Initial size: {sys.getsizeof(df)/1_000_000_000}') # print size of file (in Gigs)
    print(f'Initial shape: {df.shape}') #preview shape
    
    products = pd.Series(df['product_id'].value_counts()>10) #create bool series for whether item appears more than 10 times
    prod_list = [] #create empty list
    prod_dict = dict(products) #create dictionary of products series matching bool and product id
    for key, value in prod_dict.items():
        if value == False:
            prod_list.append(key) #make a list of just product ids from series which appear less than 10 times
    prod_in = df[df['product_id'].isin(prod_list)].index #make list of indexes of those product ids
    df.drop(index=prod_in, inplace=True) #drop those indexes (all products with less than 10 reviews)
    print(f'Size after dropping products w/reviews < 10: {sys.getsizeof(df)/1_000_000_000}') # print size of file (in Gigs)
    
    print(f'Null Preview: {df.isnull().sum()}') #preview null values
    null_perc = round((df.isnull().sum().sum()/len(df)*100),2)
    print(f'Null Percentage: {null_perc}%') #% of null value rows out of all rows
    dropped=False #null values have not been dropped
    if null_perc < .1: #automatically drop nulls if they represent less than 1%
        dropped=True
        df.dropna(inplace=True)
        print(f'Size after dropna(): {sys.getsizeof(df)/1_000_000_000}') # print size of file (in Gigs)
    else:  #if nulls are more than 1%, ask user to approve dropping
        answer = input('Would you like to drop all null values? Please enter yes or no: ') #option to drop nulls
        if answer.lower() == 'yes':
            dropped=True
            df.dropna(inplace=True)
            print(f'Size after dropna(): {sys.getsizeof(df)/1_000_000_000}') # print size of file (in Gigs)
        else:
            print('''
            As you wish...
            WARNING!
            Null values remain in Data
            ''')
    
    df['full_review'] = df['review_headline']+' '+df['review_body']#concatenate review header and body into one column for NLP
    df.drop(columns=['review_headline', 'review_body'], inplace=True)
    print(f'Size after concatenation: {sys.getsizeof(df)/1_000_000_000}') # print size of file (in Gigs)
    
    if dropped == True: #only do this if nulls have been dropped, otherwise it will break
        print('Tokenizing, lemmatizing, and removing stopwords...hold please')
        df['full_review'] = df['full_review'].map(lambda x: tokem_lite(x)) #tokenize, lemmatize, and remove stopwords
        print(f'Size after tokemmitization: {sys.getsizeof(df)/1_000_000_000}') # print size of file (in Gigs)
    
    df.to_csv(f'./data/{new_name}.csv', index=False)
    #print(f'File saved as {new_name}.csv')
    
    print(f'Final size: {sys.getsizeof(df)/1_000_000_000}') # print size of file (in Gigs)
    print(f'Final shape: {df.shape}') #preview shape
    return f'File saved as {new_name}.csv'
