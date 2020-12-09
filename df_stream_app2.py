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
    ('Overview', 'Books', 'Movies', 'Video Games')
)
@st.cache
def load_vg():
    return pd.read_pickle('./pickles/vg_look_small.pkl'), pd.read_pickle('./pickles/videog_rec.pkl')
@st.cache
def load_movies():
    return pd.read_pickle('./pickles/movies_look_small.pkl'), pd.read_pickle('./pickles/movie_rec.pkl')
@st.cache
def load_books():
    return pd.read_pickle('./pickles/books_look_small.pkl'), pd.read_pickle('./pickles/books_rec.pkl')

#movies_lookup, movies_rec = load_movies()

#vg1, vg2 = choose_look_and_rec('video games')
#movies1, movies2 = choose_look_and_rec('movies')
#books1, books2 = choose_look_and_rec('books')


def query_to_item(query, lookup):
    try:
        query=query.lower() #lowercase entry, lowercase titles (only during search, below)
        titles = list(lookup[lookup['product_title'].map(lambda x: x.lower()).str.contains(query)]['product_title'])
        return titles[0] #this is the item to search for
    except:
        return f'Sorry, "{query}" does not appear to be in the product database'

def give_recs(product):
    return dict(recommender.loc[product].sort_values()[:100]) #sort distances smallest to largest, max 100

def filter_recs(prod_dictionary, wout=''):
    if wout == '':
        top10_prods = []
        num_prod_revs = []
        avg_prod_stars = []
        for key in list(prod_dictionary.keys())[1:11]:
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
        filtered_query = [] #make empty list
        for key, value in prod_dictionary.items(): #index into dictionary of results
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

#read in pickled dataframes

#vg_df, vg_rec = pd.read_pickle('./pickles/vg_look_small.pkl'), pd.read_pickle('./pickles/videog_rec.pkl')
#movies_df, movies_rec = pd.read_pickle('./pickles/movies_look_small.pkl'), pd.read_pickle('./pickles/movie_rec.pkl')
#books_df, books_rec = pd.read_pickle('./pickles/books_look_small.pkl'), pd.read_pickle('./pickles/books_rec.pkl')

#define recommender function

    
if page == 'Overview':
    st.subheader('Overview')
    st.write('''
This model will recommend items based on Amazon product reviews written and related to products which were for sale from 1995-2015.
    ''')

    #vg1, vg2 = choose_look_and_rec('video games')
    #movies1, movies2 = choose_look_and_rec('movies')
    #books1, books2 = choose_look_and_rec('books')

elif page =='Books':
    st.subheader('Books')
    st.write('''
The book recommender was trained on 1,489,354 reviews consisting of 795,389 unique customers and 46,575 unique products.

Try it out for yourself:
    ''')
    #@st.cache
    lookup, recommender = load_books()

    query = st.text_input('Please enter a word or phrase to search": ', max_chars=50)

    search_prod = query_to_item(query, lookup)

    prod_dict = give_recs(search_prod)

    wout = st.text_input('If you would like to exclude a term from your search, please enter it here": ', max_chars=50)

    recommendation = filter_recs(prod_dict, wout)

    st.table(recommendation)


elif page =='Movies':
    st.subheader('Movies')
    st.write('''
The movie recommender was trained on 4,405,432 reviews consisting of 1,867,543 unique customers and 72,385 unique products.

Try it out for yourself:
    ''')
    #@st.cache
    lookup, recommender = load_movies()
    #lookup, recommender = movies_lookup, movies_rec

    query = st.text_input('Please enter a word or phrase to search": ', max_chars=50)

    search_prod = query_to_item(query, lookup)

    prod_dict = give_recs(search_prod)

    wout = st.text_input('If you would like to exclude a term from your search, please enter it here": ', max_chars=50)

    recommendation = filter_recs(prod_dict, wout)

    st.table(recommendation)


elif page =='Video Games':
    st.subheader('Video Games')
    st.write('''
The video game recommender was trained on 1,648,136 reviews consisting of 979,917 unique customers and 15,938 unique products.

Try it out for yourself:
    ''')

    lookup, recommender = load_vg()

    query = st.text_input('Please enter a word or phrase to search": ', max_chars=50)

    search_prod = query_to_item(query, lookup)

    prod_dict = give_recs(search_prod)

    wout = st.text_input('If you would like to exclude a term from your search, please enter it here": ', max_chars=50)

    recommendation = filter_recs(prod_dict, wout)

    st.table(recommendation)