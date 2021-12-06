import streamlit as st
import numpy as np 
import pandas as pd
import plotly.express as px
from plotly.graph_objs import *
import plotly.graph_objects as go
import plotly as py
import plotly.io as pio
import pickle
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
pio.renderers.default = 'chrome'



#st.title('Recommended for you!')
st.markdown(' <p align="center" class="big-font">  <b>Authorship Attribution <u>Check</b>   </p>', unsafe_allow_html=True)	


st.markdown("""
<style>
.big-font {
    font-size:45px !important;
}
</style>
""", unsafe_allow_html=True)

# cosine_sim = pd.read_pickle('cosine_sim.pickle')
# indices = pd.read_pickle('indices.pickle')
df1 = pd.read_pickle('df.pkl')
cosin = pd.read_pickle('cosine.pkl')
#reviews = pd.read_pickle('clean_review.pickle')
# raw = pd.read_pickle("clean_data.pickle")

df1
name = st.sidebar.text_input(''' Enter your arabic book name''')
st.sidebar.table(df1["BookTitle"])

books = df1[(df1["BookTitle"] == name)]
           #& (reviews["Polarity"] == "Positive")].reset_index(drop=True)
st.write(books)
           




# tf = TfidfVectorizer(analyzer = "word", ngram_range=(1,2), min_df=0, max_df=0.95)

# tfidf_matrix = tf.fit_transform(df1['combined_text'])

# cosine =  cosine_similarity(tfidf_matrix, tfidf_matrix)
st.write(get_recommendations(books))
st.write(cosin.to_numpy()) 
cosine= cosin.to_numpy()

def get_title_from_index(Index):
    return df1[df1.index == Index]["BookTitle"].values[0]
def get_index_from_title(BookTitle):
    return df1[df1.BookTitle == BookTitle]["index"].values[0]

def get_recommendations(book):
    book_index = get_index_from_title(book)
    similar_books = list(enumerate(cosine[book_index]))
    sortedbooks = sorted(similar_books, key = lambda x:x[1], reverse=True)[1:]
    i = 0
    for book in sortedbooks:
        print(get_title_from_index(book[0]) + " by " + df1.author[df1["index"] == book[0]])
        

        i = i+1
        if i>10:
            break
        















# def get_recommendations(name, cosine_sim, raw):
    
#     idx = indices[name]

#     sim_scores = list(enumerate(cosine_sim[idx]))

#     sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

#     sim_scores = sim_scores[1:11]


#     food_indices = [i[0] for i in sim_scores]

#     recommend = pd.DataFrame(df['Name'].iloc[food_indices]).reset_index(drop=True)
#     d = pd.merge(recommend, raw, on=None, left_on="Name", right_on="Name", how="left")
#     return d.drop(columns=["Description","Ingredients","Preparation"])
    




# def user(user_name, cosine_sim, raw):

#     user_info = reviews[(reviews["User_Name"] == user_name) & (reviews["Polarity"] == "Positive")]
    
#     if len(user_info) >= 1 :
#         food = (user_info.sample(1)).iloc[0,1]
#         return(get_recommendations(food, cosine_sim, raw))
#     else:
#         food = (df.sample(1)).iloc[0,1]
#         return(get_recommendations(food, cosine_sim, raw))


# recommended = user(name, cosine_sim, raw)
# recommended.sort_values("Rating", ascending=False, inplace=True)
# recommended = recommended.reset_index(drop=True)

# recom = recommended.sort_values("Rating", ascending=False)

    
# fig = px.bar(recom, x='Name', y='Rating', color='Name',color_discrete_sequence=px.colors.diverging.Geyser, height=600, width=900)
    
# fig.update_xaxes(showgrid=False)
# fig.update_yaxes(showgrid=False)
    
    
# fig.update_layout(template="plotly_white",xaxis_showgrid=False, yaxis_showgrid=False)
    
# fig.update_traces( marker_line_color='rgb(8,48,107)',
#                   marker_line_width=2, opacity=0.6)
    
# fig.update_layout(showlegend=False, title="Rating",
# xaxis_title="Recommended Recipes",
# yaxis_title="Rate")

    
# fig.update_xaxes(showline=True, linewidth=1, linecolor='black')
# fig.update_yaxes(showline=True, linewidth=1, linecolor='black')
                               
# st.table(recommended)

# st.plotly_chart(fig)








                               
