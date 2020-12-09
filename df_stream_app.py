import pickle
import streamlit as st
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
#https://docs.streamlit.io/en/stable/api.html#display-data

st.set_page_config(
    page_icon=':books:',
    initial_sidebar_state='auto'
)

st.title('Find your next adventure...')

page = st.sidebar.selectbox(
    'Select-A-Page',
    ('Overview', 'Books', 'Movies', 'Video Games', 'Recommender')
)

#read in pickled dataframes

vg_df, vg_rec = pd.read_pickle('./pickles/vg_look_small.pkl'), pd.read_pickle('./pickles/videog_rec.pkl')
movies_df, movies_rec = pd.read_pickle('./pickles/movies_look_small.pkl'), pd.read_pickle('./pickles/movie_rec.pkl')
books_df, books_rec = pd.read_pickle('./pickles/books_look_small.pkl'), pd.read_pickle('./pickles/books_rec.pkl')

#define recommender function

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
        query=query.lower() #lowercase entry, lowercase titles (only during search, below)
        titles = list(lookup[lookup['product_title'].map(lambda x: x.lower()).str.contains(query)]['product_title'])
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
            
            wout = wout.lower() #capitlize first letters
            query_dict = dict(recommender.loc[q].sort_values())
            filtered_query = [] #make empty list
            for key, value in query_dict.items(): #index into dictionary of results
                if wout not in key.lower(): #check if avoided keyword is in results
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

#read in pickled dataframes

#vg_df, vg_rec = pd.read_pickle('./pickles/vg_look_small.pkl'), pd.read_pickle('./pickles/videog_rec.pkl')
#movies_df, movies_rec = pd.read_pickle('./pickles/movies_look_small.pkl'), pd.read_pickle('./pickles/movie_rec.pkl')
#books_df, books_rec = pd.read_pickle('./pickles/books_look_small.pkl'), pd.read_pickle('./pickles/books_rec.pkl')

#define lookup function(s)
#@st.cache

    
if page == 'Overview':
    st.subheader('Overview')
    st.write('''
This model will recommend items based on Amazon product reviews written and related to products which were for sale from 1995-2015.
    ''')

elif page =='Books':
    st.subheader('Books')
    st.write('''
The book recommender was trained on 1,489,354 reviews consisting of 795,389 unique customers and 46,575 unique products.

Here are the most common words in book reviews sorted by star-rating, as extracted using a count-vecotrizer in conjunction with a custom stop-word list:
    ''')

elif page =='Movies':
    st.subheader('Movies')
    st.write('''
The movie recommender was trained on 4,405,432 reviews consisting of 1,867,543 unique customers and 72,385 unique products.

Here are the most common words in movie reviews sorted by star-rating, as extracted using a count-vecotrizer in conjunction with a custom stop-word list:
    ''')

elif page =='Video Games':
    st.subheader('Video Games')
    st.write('''
The video game recommender was trained on 1,648,136 reviews consisting of 979,917 unique customers and 15,938 unique products.

Here are the most common words in video game reviews sorted by star-rating, as extracted using a count-vecotrizer in conjunction with a custom stop-word list:
    ''')

elif page =='Recommender':

    category = st.text_input('Please choose "books", "movies" or "video games": ', max_chars=50)

    query = st.text_input('Please enter a word or phrase to search": ', max_chars=50)

    wout = st.text_input('If you would like to exclude a term from your search, please enter it here": ', max_chars=50)

    if wout == '':
        recommendation = rec_search(category, query)
    else:
        recommendation = rec_search(category, query, wout)

    #pd.set_option('display.max_colwidth', -1)

    if type(recommendation) == str: #if outcome is a string, return a string
        st.write(f'Recommendations: {recommendation}')
    else: #otherwise, return a dataframe
        #pd.set_option('display.max_colwidth', -1)
        #st.write(f'Recommendations: {st.table(recommendation)}')
        st.table(recommendation)