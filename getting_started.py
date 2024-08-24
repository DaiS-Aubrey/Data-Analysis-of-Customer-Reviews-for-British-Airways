#!/usr/bin/env python
# coding: utf-8

# In[3]:


import requests
from bs4 import BeautifulSoup
import pandas as pd


# In[4]:


base_url = "https://www.airlinequality.com/airline-reviews/british-airways"
pages = 10
page_size = 100

reviews = []

# for i in range(1, pages + 1):
for i in range(1, pages + 1):

    print(f"Scraping page {i}")

    # Create URL to collect links from paginated data
    url = f"{base_url}/page/{i}/?sortby=post_date%3ADesc&pagesize={page_size}"

    # Collect HTML data from this page
    response = requests.get(url)

    # Parse content
    content = response.content
    parsed_content = BeautifulSoup(content, 'html.parser')
    for para in parsed_content.find_all("div", {"class": "text_content"}):
        reviews.append(para.get_text())
    
    print(f"   ---> {len(reviews)} total reviews")


# In[5]:


df = pd.DataFrame()
df["reviews"] = reviews
df.head()


# In[6]:


df.to_csv("/Users/shuangtongdai/Desktop/forage/BA_reviews.csv")


# In[13]:


#data cleaning
f = pd.read_csv('BA_reviews.csv')
df = df[df['reviews'].str.contains('Trip Verified', na=False)]
df['reviews'] = df['reviews'].str.replace('✅ Trip Verified |', '', regex=False)
df.head()


# In[21]:


df['reviews'] = df['reviews'].str.replace(r'[^\w\s]', '', regex=True).str.strip()
df['reviews'] = df['reviews'].str.lower()
df.head(20)


# In[27]:


import nltk
nltk.download('stopwords')
nltk.download('punkt')

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

stop_words = set(stopwords.words('english'))

def remove_stop_words(text):
    word_tokens = word_tokenize(text)
    filtered_text = [word for word in word_tokens if word.lower() not in stop_words]
    return ' '.join(filtered_text)

df['reviews'] = df['reviews'].apply(remove_stop_words)

df.head(20)


# In[31]:


#text analysis
get_ipython().system('pip install wordcloud')
from wordcloud import WordCloud
import matplotlib.pyplot as plt

text = ' '.join(df['reviews'])
wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text)

plt.figure(figsize=(10, 5))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.show()


# In[35]:


get_ipython().system('pip install textblob')
from textblob import TextBlob

def get_sentiment(text):
    analysis = TextBlob(text)
    return analysis.sentiment.polarity

df['sentiment'] = df['reviews'].apply(get_sentiment)


# In[37]:


import seaborn as sns

plt.figure(figsize=(10, 5))
sns.histplot(df['sentiment'], bins=30, kde=True)
plt.title('Sentiment Distribution of Reviews')
plt.xlabel('Sentiment Score')
plt.ylabel('Frequency')
plt.show()


# In[41]:


from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation

vectorizer = CountVectorizer(max_df=0.95, min_df=2, stop_words='english')
X = vectorizer.fit_transform(df['reviews'])

lda = LatentDirichletAllocation(n_components=5, random_state=45)
lda.fit(X)

def print_top_words(model, feature_names, n_words=10):
    for topic_idx, topic in enumerate(model.components_):
        print(f"Topic #{topic_idx}:")
        print(" ".join([feature_names[i] for i in topic.argsort()[:-n_words - 1:-1]]))
    print()

print_top_words(lda, vectorizer.get_feature_names_out())


# In[ ]:




