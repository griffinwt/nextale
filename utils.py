#imports
import pandas as pd
import numpy as np
import requests
from bs4 import BeautifulSoup, NavigableString, Tag
import re
import time
import random
import sys
pd.set_option('display.max_colwidth', None)

from sklearn.metrics.pairwise import pairwise_distances, cosine_distances, cosine_similarity
from scipy import sparse

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
    slem = [lemmatizer.lemmatize(word) for word in stok]
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
    
    print(f'Final size: {sys.getsizeof(df)/1_000_000_000}') # print size of file (in Gigs)
    print(f'Final shape: {df.shape}') #preview shape
    return f'File saved as {new_name}.csv'



def make_recommender_df(df, name): #converts df to recommender and saves under input name
    #drop any null values remaining from cleaning (will only be a handful in concatenated NLP column)
    df.dropna(inplace=True)
    #ensure dates column is in date-time format
    df['review_date'] = pd.to_datetime(df['review_date'])
    #make T/F list for if cs has written more than 1 review for the same product
    review_bools = df.groupby('customer_id')['product_id'].value_counts()>1
    
    #list of customer id numbers for those with more than 1 review for same item
    xtra_rev_cs = []
    for key, value in dict(review_bools[review_bools==True]).items():
        xtra_rev_cs.append({key[0]:key[1]}) #return customer id and product id ONLY
    
    #make list of original df indexes corresponding to reviews that need to be dropped
    rev_indexes_to_drop = []
    for pair in xtra_rev_cs:
        for key, value in pair.items():
            rev_indexes_to_drop.append(  #add index numbers to the empty list
            df[(df['customer_id'] == key) &    #where customer id is the key from xtra_rev_cs...
           (df['product_id']==value)].sort_values( #and product id is the value from xtra_rev_cs
            by='review_date', ascending=False).index[1:] #starting with the SECOND index number
            )
    ritd_2 = [] #list for indexes
    for n in rev_indexes_to_drop:
        for k in n:
            ritd_2.append(k)
    print(f'Dropping {len(ritd_2)} duplicate values.') #print status update for number of duplicates being dropped
    df.drop(index=ritd_2, inplace=True) #drop all index numbers in list ritd_2
    
    #make new dataframe for recommender build
    df2 = df[['customer_id', 'product_id', 'product_title', 'star_rating']].copy()
    print(f"Unique customers: {df2['customer_id'].nunique()}") #preview number of unique customers
    print(f"Unique products: {df2['product_id'].nunique()}") #preview number of unique products
    
    unique_prods = list(set(df2['product_title'])) #create list of unique products
    prod_index = {p:i for i,p in enumerate(unique_prods)} #match unique products with integer values
    df2['prod_numerical'] = df2['product_title'].apply(lambda x: prod_index[x]) #add column to df2
    
    unique_cs = list(set(df2['customer_id'])) #create list of unique customers
    cs_index = {p:i for i,p in enumerate(unique_cs)} #match unique customers with (smaller) integer values
    df2['cs_numerical'] = df2['customer_id'].apply(lambda x: cs_index[x]) #add column to df2
    
    df2['star_rating'] = df2['star_rating'].astype(np.int8) #convert to take up less memory
    
    #create sparse matrix comparing customer ratings and products
    sparse_reviews = sparse.csr_matrix((df2.star_rating, (df2.prod_numerical, df2.cs_numerical)), dtype=np.int8)
    print(f'Size of matrix: {sparse_reviews.shape}') #preview size of sparse matrix
    #get cosine distances between items
    dists = pairwise_distances(sparse_reviews, metric='cosine')
    
    #create recommender df
    
    print('Making df dictionary...')
    rec_cols = unique_prods
    rec_dict = {}
    for n in range(len(dists)):  #create dictionary to use in creating sparse data frame
        rec_dict[rec_cols[n]] = pd.arrays.SparseArray(dists[n], fill_value=1) #do not store values of 1
    print('Dictionary made - sparse dataframe under construction...')
    recommender_df_sparse = pd.DataFrame(rec_dict,
                             index=rec_cols)
    
    #recommender_df = pd.DataFrame(dists,   #old way (used too much memory)
                             #index=unique_prods,
                             #columns=unique_prods)
    
                  
    print(f'Size of {name} Recommender df: {sys.getsizeof(recommender_df_sparse)/1_000_000_000} GB') #check size in GB
    
    #pickle df
    recommender_df_sparse.to_pickle(f'./pickles/{name}.pkl')
    #return recommender_df
    return f'Recommender DataFrame for {name} pickled successfully as ./pickles/{name}.pkl'



def size_in_gb(some_df): #returns size of input value in gigabytes
    return f'{sys.getsizeof(some_df)/1_000_000_000} GB'




def make_smaller_lookup(lookup_df):
    print(lookup_df.shape) #preview initial shape
    print(lookup_df['product_title'].nunique()) #unique product count
    lookup_df = lookup_df.groupby('product_title').first().copy() #only 1 row for each unique product
    lookup_df.reset_index(inplace=True) #add index, was removed during groupby
    lookup_df.drop(columns=['customer_id', 'product_id', 'star_rating', 'review_date'], inplace=True) #drop unused columns
    lookup_df.sort_values(by='tot_prod_reviews', ascending=False, inplace=True) #sort by most reviews
    lookup_df.reset_index(inplace=True) #reset index
    lookup_df.drop(columns=['index'], inplace=True) #drop original index (needed previously in order to sort)
    return lookup_df