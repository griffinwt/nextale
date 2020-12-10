import pickle
import streamlit as st
import numpy as np
import pandas as pd
import bz2
import _pickle as cPickle
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

#recommender function

def make_recs_new(query, wout=''):  #need to set lookup and recommender global variables prior to calling   
    try:
        query=query.lower() #lowercase entry, lowercase titles (only during search, below)
        titles = list(lookup[lookup['product_title'].map(lambda x: x.lower()).str.contains(query)]['product_title'])
        q = titles[0] #this is the item to search for

        message = f'''
        **Most Popular Item Containing Your Search Term(s):** {q}  
        There are {round(lookup[lookup['product_title']==q]['tot_prod_reviews'].mean())} total reviews for this item and it has an average star rating of {round(lookup[lookup['product_title']==q]['avg_prod_stars'].mean(), 2)}
        '''   

        if wout == '':           
            top10_prods = []
            num_prod_revs = []
            avg_prod_stars = []
            for key in list(recommender[q].keys())[1:11]:
                top10_prods.append(key)
                num_prod_revs.append(round(lookup[lookup['product_title']==key]['tot_prod_reviews'].mean()))
                avg_prod_stars.append(round(lookup[lookup['product_title']==key]['avg_prod_stars'].mean(), 2))
            final_output_df = pd.DataFrame(data = {
                'Recommended Items':top10_prods,
                'Total Reviews for Product':num_prod_revs,
                'Avg Product Star Rating(1-5)':avg_prod_stars
            }, index=range(1,11))
            return message, final_output_df            
            
        else:
            
            wout = wout.lower() #lowercase
            filtered_query = [] #make empty list
            for key in list(recommender[q].keys()):
                if wout not in key.lower(): #check if avoided keyword is in results
                    filtered_query.append(key)
            top10_prods = []
            num_prod_revs = []
            avg_prod_stars = []
            for item in filtered_query[1:11]:
                top10_prods.append(item)
                num_prod_revs.append(round(lookup[lookup['product_title']==item]['tot_prod_reviews'].mean()))
                avg_prod_stars.append(round(lookup[lookup['product_title']==item]['avg_prod_stars'].mean(), 2))
            final_output_df = pd.DataFrame(data = {
                'Recommended Items':top10_prods,
                'Total Reviews for Product':num_prod_revs,
                'Avg Product Star Rating(1-5)':avg_prod_stars
                }, index=range(1,11))
            return message, final_output_df
        
    except:
        return 'Error', f'Sorry, "{query}" does not appear to be in the product database'

    
if page == 'Overview':
    st.subheader('Overview')
    st.write('''

Recommendation systems are ubiquitous in today's attention-driven online environment, whether used by an online retailer to recommend similar products or a media application to prompt binge-watching. These systems use a combination of well known and proprietary techniques to engage customers with customized recommendations based on their previous preference. In this project, I created a relatively simple version of one of these recommenders using customer reviews between 1995-2015 provided by Amazon. I specifically looked at products in the categories of books, movies, and video games. To reflect the narrative element in each of these products, I titled my project "Nextale". It is a tool to assist the user in finding their "next tale" based on something they enjoyed previously.

''')

    st.image('./images/sample_vectors.png')

elif page =='Books':
    st.subheader('Books')
    st.write('''
The book recommender was built from 1,489,354 reviews consisting of 795,389 unique customers and 46,575 unique products.

Try it out for yourself:
    ''')

    lookup = pd.read_pickle('./compressed/books_look_p3')

    with open('./compressed/new_books_rec.pkl', 'rb') as f:
        recommender = pickle.load(f)

    query = st.text_input('Please enter a word or phrase to search: ', max_chars=50)

    wout = st.text_input('If you would like to exclude a term from your search, please enter it here: ', max_chars=50)

    searched, recommendation = make_recs_new(query, wout)

    if type(recommendation) == str:
        st.write(recommendation) #if result is a string, print it
    else:
        st.write(searched) #show searched term
        st.table(recommendation) #if result is a df, show it

    st.write('''
    *in order to expedite search results and save memory, this application works off of a truncated version of the full recommender; for each product, the 100 most similar products are stored in memory. This means that theoretically, if a user's "exclusion" search term is too broad, less than 10 similar products may be available which would compromise the resulting printout. While possible, this error has never been observed in any tests of the application to date.
    ''')


elif page =='Movies':
    st.subheader('Movies')
    st.write('''
The movie recommender was built from 4,405,432 reviews consisting of 1,867,543 unique customers and 72,385 unique products.

Try it out for yourself:  
*Please note that the movies product list is very large - loading may take a few extra seconds!*
    ''')

    lookup = pd.read_pickle('./compressed/movies_look_p3')

    recommender = bz2.BZ2File('./compressed/movies_rec_c.pbz2')
    recommender = cPickle.load(recommender)

    query = st.text_input('Please enter a word or phrase to search: ', max_chars=50)

    wout = st.text_input('If you would like to exclude a term from your search, please enter it here: ', max_chars=50)

    searched, recommendation = make_recs_new(query, wout)

    if type(recommendation) == str:
        st.write(recommendation) #if result is a string, print it
    else:
        st.write(searched) #show searched term
        st.table(recommendation) #if result is a df, show it

    st.write('''
    *in order to expedite search results and save memory, this application works off of a truncated version of the full recommender; for each product, the 100 most similar products are stored in memory. This means that theoretically, if a user's "exclusion" search term is too broad, less than 10 similar products may be available which would compromise the resulting printout. While possible, this error has never been observed in any tests of the application to date.
    ''')

elif page =='Video Games':
    st.subheader('Video Games')
    st.write('''
The video game recommender was built from 1,648,136 reviews consisting of 979,917 unique customers and 15,938 unique products.

Try it out for yourself:
    ''')

    lookup = pd.read_pickle('./compressed/vg_look_p3')

    with open('./compressed/new_vg_rec.pkl', 'rb') as f:
        recommender = pickle.load(f)

    query = st.text_input('Please enter a word or phrase to search: ', max_chars=50)

    wout = st.text_input('If you would like to exclude a term from your search, please enter it here: ', max_chars=50)

    searched, recommendation = make_recs_new(query, wout)

    if type(recommendation) == str:
        st.write(recommendation) #if result is a string, print it
    else:
        st.write(searched) #show searched term
        st.table(recommendation) #if result is a df, show it

    st.write('''
    *in order to expedite search results and save memory, this application works off of a truncated version of the full recommender; for each product, the 100 most similar products are stored in memory. This means that theoretically, if a user's "exclusion" search term is too broad, less than 10 similar products may be available which would compromise the resulting printout. While possible, this error has never been observed in any tests of the application to date.
    ''')