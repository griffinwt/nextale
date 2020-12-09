- [Problem Statement](#Problem-Statement)
- [Summary](#Summary)
- [Next Steps](#Next-Steps)
- [Data Dictionary](#Data-Dictionary)
- [External Resources](#External-Resources)

---

## Problem Statement

Recommendation systems are ubiquitous in today's attention-driven online environment, whether used by an online retailer to recommend similar products or a media application to prompt binge-watching. These systems use a combination of well known and proprietary techniques to engage customers with customized recommendations based on their previous preference. In this project, I created a relatively simple version of one of these recommenders using customer reviews between 1995-2015 provided by Amazon. I specifically looked at products in the categories of books, movies, and video games. To reflect the narrative element in each of these products, I titled my project "Nextale". It is a tool to assist the user in finding their "next tale" based on something they enjoyed previously.

---

## Summary

My first task in undertaking this project was finding a suitable volume and breadth of information commensurate with my goal. This presented itself in the form of the "Amazon Customer Review Dataset" provided by Amazon Web Services. There are reportedly over 130 million reviews available, but as stated above I limited the scope of my project to three segments - books, movies, and video games. I wrote a custom function to automatically clean / organize the initial raw file and re-save it. Among other processes, this function dropped null values (representing less than 1% of all data), unused columns, and any item with less than 10 reviews. It also concatenated the review title and review text into one string, and then tokenized, lemmatized, and removed-stop-words from that combined review text so that the remainder was ripe for NLP. Upon completion, I had (NEED#) book reviews, (NEED#) movie reviews, and (NEED#) video game reviews.

(picture of custom python function readout of one cleaning cycle)

<img src="./Chapter_1/images/top_5_reasons.png" width="75%" height="75%">

Having cleaned the review data, I set out to create a cosine distance recommender model. This model finds the similarity between products by measuring the distance between their vectors created from review grades by customers who reviewed both products.

(visualization of simple cosine distance)

As seen in the visualization above, we can see that film 1 and film 2 are close related by their review scores, meaning that someone who like film one would have film 2 recommended to them. This is a microcosm of how my recommender works. One potential hurdle in this form of recommender is if a reviewr has made more than one review for the same product. While this represented a relatively small proportion of my total reviews, I am generally reluctant to remove data if there is some way to salvage it and use it. In my exploration, I found that some of these reviews were simply duplicates (the reviewer presumably revisited the "submit" button in quick and aggressive fashion following their entry), but others were people who had one initial reaction and then later returned to give a second opinion after more evaluation. Without having the time or desire to examine each one of these instances individually, I settled upon sorting them by date and keeping the most recent entry. For customers who accidentally submitted their review multiple times at the same instant, any one of them would be a fair enough representation of their opinion, and for those who made an initial and then subsequent contribution, their final opinion would be the one that counted.

Again, I wrote a custom python function to transform the initial pandas dataframe of review information into a recommender matrix. The final output was a product x product dataframe with the numbers inside representing the cosine distance between the respective products. A "0" would be where a product interescted with itself, and a 1 would be where two products had no relationship whatsoever. These recommender dataframes initially presented a memory challenge - for example, the largest (movies) is over 72,000 x 72,000 or over 5,184,000,000 cells! This initially took up over 40 GB of space, much more than I could hope to hold in memory at one time. I overcame this obstacle by encoding a "sparse" transformer into my python function. By default, the sparse representation tells python not to save any zeroes to memory - this is helpful when the majority of a large dataframe contents are zeroes. By not storing those integers, a significant amount of memory is saved and as long as all *other* values are tangible, we (and python) can infer that any "empty" values would be zeroes. I realized, however, that the majority of my recommender dataframe's contents were, in fact, "1"s. With such a large volume of products, most would be unrelated to each other because they would not share a common reviewer. With that in mind, I adjusted the sparse function to not remember any of the 1's in my dataframe. This cut my over 40 GB dataframe to less than 2 GB!

(before and after rec df creation screenshots?)

<img src="./Chapter_2/Images/top_10_models.png" width="75%" height="75%">

With all three product x product dataframes created, I was finally able to make recommendations! I built a large function that accepted paramaters for category, search term, and (optional) filter-out term, and returned the top 10 most similar items, as well as those items' total number of reviews and average star rating. Making use of the NLP framework I set up when cleaning, I also pulled out the five most common recurring terms in reviews for each item and returned those as well, to give users another angle from which to perceive the aggregate opinions of reviewers with regards to the recommended items.

Not content with having a model that only ran in my Jupyter notebook, I set a stretch goal of deploying my recommender online for others to experience it. After experimenting with Flask, I settled on the popular streamlit.io python module. This allowed me to create a clean yet modern-looking web app without requiring the html/css knowledge that Flask would to achieve a similarly polished look.

The largest recurring challenge that I encountered during this web deployment phase was simply the size of my data. Each of the three categories requrired two dataframes: a "lookup" dataframe to run the search query through and return information about products (number of reviews, average star rating), and a "recommender" dataframe which was my aforementioned product x product matrix. In total, this meant I had to read in six dataframes, the largest of which was still almost 2 GB. So while my recommender app worked, it was tediously slow.

To overcome this, I used my recommender systems to create dictionaries which stored the top 100 most similar products to each other product. For example, my 72k x 72k product dataframe became a dictionary with 72k entries, each having a list of 100 titles associated with it. The potential downside to this method was that with only 100 similar items, there was always a theoretical possibility that one of the "filter-out" keywords a user entered might apply to more than 90 of the returned products, leaving the recommender with less than 10 items to recommend. In practice, I was unable to trigger this sort of error and hopefully no one ever does! The upside, however, far outweighed the potential downside, in the sense that my search returns became reliably rapid.


<img src="./Chapter_3/am_by_year.png" width="75%" height="75%">

(text)


---

## Next Steps

- Because I used a systematic, function-based approach, my cleaning and recommender creation processes would easily generalize to other categories of the Amazon Review dataset. With more time and less memory constraints, I could easily expand my recommender to encompass other types of goods.
- The Natural Language Processing I performed in this project was somewhat shallow as it only contributed to a tangental feature; with more time, I believe I could glean greater insights comparing user review text, combining it either by star rating, product, average product star rating, or by similar products.
- As the common refrain goes, more data would always be better; I would greatly enjoy introducing more recent reviews (post-2015) to expand my model in both product width and total review volume.

---

## Data Dictionary

|Feature|Type|Description|
|---|---|---|
|**event_id**|str|unique accident ID|
|**investigation_type**|str|categorical; Accident or Incident|
|**event_date**|time object|month/day/year of occurrence|
|**location**|str|City, State of occurrence|


---

### External Resources
https://s3.amazonaws.com/amazon-reviews-pds/readme.html
https://www.dummies.com/web-design-development/site-development/how-to-create-a-drop-down-list-in-an-html5-form/

