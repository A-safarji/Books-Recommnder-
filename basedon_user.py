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
st.set_option('deprecation.showPyplotGlobalUse', False)
#st.set_page_config(layout="wide")

#st.title('Recommended for you!')
st.markdown(' <p align="center" class="big-font">  <b>Authorship Attribution <u> 🌟 T5 🇸🇦</b>   </p>', unsafe_allow_html=True)	


st.markdown("""
<style>
.big-font {
    font-size:50px !important;
}
</style>
""", unsafe_allow_html=True)



st.markdown("""
إسناد التأليف العربي هو مهمة البحث عن مؤلف المستند. لتحقيق هذا الغرض ، يقارن المرء نص الاستعلام بنموذج المؤلف المرشح ويحدد احتمال نموذج الاستعلام.

Arabic authorship attribution is the task of finding the author of a document.
To achieve this purpose, one compares a query text with a model of the candidate author and determines the likelihood of the model for the query.
	""")

raw_text = st.text_area("Authorship Attribution Check","Enter Text Here")

df1 = pd.read_pickle('df.pkl')
cosin = pd.read_pickle('cosine.pkl')



if st.checkbox("Show orignal dataframe | عرض جميع الكتب الموجودة بنظام التوصية"):
	dataframe=df1
	#dataframe.drop('Unnamed: 0', axis=1, inplace=True)
	dataframe



st.sidebar.header('نظام التوصية')
name = st.sidebar.text_input(''' ادخل اسم الكتاب''')
st.sidebar.write(''' Our Books Collections Below:''')
st.sidebar.write(''' 
مجموعات كتبنا أدناه''')

st.sidebar.table(df1["BookTitle"])

st.write('---')

st.subheader('Your Selected Book Title Details | تفاصيل عنوان الكتاب المختار ')
books = df1[(df1["BookTitle"] == name)]
           #& (reviews["Polarity"] == "Positive")].reset_index(drop=True)
st.write(books)
           
st.write('---')

#st.write(cosin.to_numpy()) 
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
        st.write(" Title: "+ get_title_from_index(book[0]) + " ♦️ "  + " Author: " + df1.author[df1["index"] == book[0]])
      
        i = i+1
        if i>10:
            break
   
#df1.BookTitle[df1["index"] == book[0]]  
st.subheader('💡 Your Recommended Books | كتبك الموصى بها ')
try:
	st.write(get_recommendations(name))
except:
        st.error("🔴 Please make sure that you only enter a name of your book | يرجى التأكد من إدخال اسم كتابك فقط")
        st.stop()

st.write('---')

st.header('Books in Recommnder System')

fig = px.bar(df1, x='author' , y='text_length', color='author',color_discrete_sequence=px.colors.diverging.Geyser, height=600, width=900)
    
fig.update_xaxes(showgrid=False)
fig.update_yaxes(showgrid=True)
    
    
fig.update_layout(template="plotly_white",xaxis_showgrid=False, yaxis_showgrid=False)
    
fig.update_traces( marker_line_color='rgb(8,48,107)',
                  marker_line_width=2, opacity=0.6)
    
fig.update_layout(showlegend=True, title="Books Word Count",
xaxis_title="Author Name",
yaxis_title="Text Length")

    
fig.update_xaxes(showline=True, linewidth=1, linecolor='black')
fig.update_yaxes(showline=True, linewidth=1, linecolor='black')
                               
#st.table(get_recommendations(name))

st.plotly_chart(fig)





st.write('---')
st.write('## Contact Our Group')


st.write("""
[Authorship Attribution](https://github.com/A-safarji) - feel free to contact!
""")
                               
