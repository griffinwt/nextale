import pickle
import streamlit as st
import numpy as np
import pandas as pd
import re
import time
import random
import sys
from sklearn.metrics.pairwise import pairwise_distances, cosine_distances, cosine_similarity
from scipy import sparse
from matplotlib import pyplot as plt
import utils as ut

st.title('Recommender System')

#read in pickled dataframes
books_df, books_rec, movies_df, movies_rec, vg_df, vg_rec = pd.read_pickle('./pickles/books_look.pkl'), pd.read_pickle('./pickles/books_rec.pkl'), pd.read_pickle('./pickles/movies_look.pkl'), pd.read_pickle('./pickles/movie_rec.pkl'), pd.read_pickle('./pickles/videog_look.pkl'), pd.read_pickle('./pickles/videog_rec.pkl')

#define lookup function

def rec_search(category, query, wout='no'):

    if category.lower() == 'video games':
        lookup, recommender = vg_df, vg_rec #pd.read_pickle('./pickles/videog_look.pkl'),pd.read_pickle('./pickles/videog_rec.pkl')
    elif category.lower() == 'movies':
        lookup, recommender = movies_df, movies_rec #pd.read_pickle('./pickles/movies_look.pkl'), pd.read_pickle('./pickles/movie_rec.pkl') 
    elif category.lower() == 'books':
        lookup, recommender = books_df, books_rec #pd.read_pickle('./pickles/books_look.pkl'), pd.read_pickle('./pickles/books_rec.pkl')
    else:
        return "Sorry, that wasn't one of the available categories"

    try:
        query=query.title() #uppercase first letter of each word in the query in case it's not entered that way
        titles = list(lookup[lookup['product_title'].str.contains(query)]['product_title'])
        q = titles[0] #this is the item to search for

        if wout.lower() == 'no':
            query_dict = dict(recommender.loc[q].sort_values())
            
            final_printout = ''
            for key in list(query_dict.keys())[1:11]:
                final_printout += '\n' #add new line
                final_printout += f'{key}' #print item name
                final_printout += '\n'
                final_printout += f"""
                This item has {round(lookup[lookup['product_title']==key]['tot_prod_reviews'].mean())} reviews 
                and a {round(lookup[lookup['product_title']==key]['avg_prod_stars'].mean(), 2)} average star rating"""
                final_printout += '\n'

            return (f'''

            Recommending items similar to: {q}
            This item has {round(lookup[lookup['product_title']==q]['tot_prod_reviews'].mean())} reviews
            and a {round(lookup[lookup['product_title']==q]['avg_prod_stars'].mean(), 2)} average star rating.

            Here are the 10 recommended items for you based on your search parameters:
            
            {final_printout}
            
            ''')
            
        else:
            
            wout = wout.lower() #lowercase
            query_dict = dict(recommender.loc[q].sort_values())
            filtered_query = [] #make empty list
            for key, value in query_dict.items(): #index into dictionary of results
                if wout not in key.lower(): #check if avoided keyword is in results
                    filtered_query.append((key, value)) #make list of results that DON'T include "wout" keyword
            
            final_printout = ''
            for item in filtered_query[1:11]:
                final_printout += '\n' #add new line
                final_printout += f'{item[0]}' #print item name
                final_printout += '\n'
                final_printout += f"""
                This item has {round(lookup[lookup['product_title']==item[0]]['tot_prod_reviews'].mean())} reviews 
                and a {round(lookup[lookup['product_title']==item[0]]['avg_prod_stars'].mean(), 2)} average star rating"""
                final_printout += '\n'
            return (f'''

            Recommending items similar to: {q}
            This item has {round(lookup[lookup['product_title']==q]['tot_prod_reviews'].mean())} reviews
            and a {round(lookup[lookup['product_title']==q]['avg_prod_stars'].mean(), 2)} average star rating.

            Here are the 10 recommended items for you based on your search EXCLUDING "{wout}":
            
            {final_printout}
            
            ''')
        
    except:
        return (f'Sorry, "{query}" does not appear to be in the product database')

def rec_search_df(category, query, wout='no'):

    if category.lower() == 'video games':
        lookup, recommender = vg_df, vg_rec #pd.read_pickle('./pickles/videog_look.pkl'),pd.read_pickle('./pickles/videog_rec.pkl')
    elif category.lower() == 'movies':
        lookup, recommender = movies_df, movies_rec #pd.read_pickle('./pickles/movies_look.pkl'), pd.read_pickle('./pickles/movie_rec.pkl') 
    elif category.lower() == 'books':
        lookup, recommender = books_df, books_rec #pd.read_pickle('./pickles/books_look.pkl'), pd.read_pickle('./pickles/books_rec.pkl')
    else:
        return "Sorry, that wasn't one of the available categories"

    try:
        query=query.title() #uppercase first letter of each word in the query in case it's not entered that way
        titles = list(lookup[lookup['product_title'].str.contains(query)]['product_title'])
        q = titles[0] #this is the item to search for

        if wout.lower() == 'no':
            query_dict = dict(recommender.loc[q].sort_values())

            top10_prods = []
            num_prod_revs = []
            avg_prod_stars = []
            for key in list(query_dict.keys())[1:11]:
                top10_prods.append(key)
                num_prod_revs.append(round(lookup[lookup['product_title']==key]['tot_prod_reviews'].mean()))
                avg_prod_stars.append(round(lookup[lookup['product_title']==key]['avg_prod_stars'].mean(), 2))
            #print(top10_prods, num_prod_revs, avg_prod_stars)
            final_output_df = pd.DataFrame(data = {
                'Recommended Items':top10_prods,
                'Total Reviews for Product':num_prod_revs,
                'Avg Product Star Rating(1-5)':avg_prod_stars
            }, index=range(1,11))

            return final_output_df            



        else:

            wout = wout.title() #capitlize first letters
            query_dict = dict(recommender.loc[q].sort_values())
            filtered_query = [] #make empty list
            for key, value in query_dict.items(): #index into dictionary of results
                if wout not in key: #check if avoided keyword is in results
                    filtered_query.append((key, value)) #make list of results that DON'T include "wout" keyword

            top10_prods = []
            num_prod_revs = []
            avg_prod_stars = []
            for item in filtered_query[1:11]:
                top10_prods.append(item[0])
                num_prod_revs.append(round(lookup[lookup['product_title']==item[0]]['tot_prod_reviews'].mean()))
                avg_prod_stars.append(round(lookup[lookup['product_title']==item[0]]['avg_prod_stars'].mean(), 2))


            final_output_df = pd.DataFrame(data = {
                'Recommended Items':top10_prods,
                'Total Reviews for Product':num_prod_revs,
                'Avg Product Star Rating(1-5)':avg_prod_stars
                }, index=range(1,11))



            return final_output_df

    except:
        return f'Sorry, "{query}" does not appear to be in the product database'
    

category = st.text_input('Please choose "books", "movies" or "video games": ', max_chars=50)

query = st.text_input('Please enter a word or phrase to search": ', max_chars=50)

wout = st.text_input('If you would like to exclude a term from your search, please enter it here": ', max_chars=50)

if wout == '':
    recommendation = rec_search(category, query)
else:
    recommendation = rec_search(category, query, wout)

st.write(f'Recommendations: {recommendation}')