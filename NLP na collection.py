#!/usr/bin/env python
# coding: utf-8

# In[1]:


# normalizar texto (remoção de código, por exemplo espaços: \xa0) e remover tags
# transformar as letras em lowercase
# remoção de pontuação
# remover números
# remover stopwords
# lemmatization


# In[2]:


import pymongo
from pymongo import MongoClient
import pandas as pd


# In[3]:


client = MongoClient('mongodb+srv://dev_data_science:Pmr129568wEuMz9j@nodisatlasgcpiowa0001-s97vy.gcp.mongodb.net/data_science?retryWrites=true&w=majority')
db = client.data_science


# In[4]:


# db.list_collection_names()


# In[5]:


cursor_sku_seller = db.sku_seller.find()
cursor_seller = db.seller.find()
# cursor_product_seller = db.product_seller.find()
# cursor_category = db.category.find()


# In[6]:


df_sku_seller = pd.DataFrame(list(cursor_sku_seller))
df_seller = pd.DataFrame(list(cursor_seller))
# df_product_seller = pd.DataFrame(list(cursor_product_seller))
# df_category = pd.DataFrame(list(cursor_category))


# In[7]:


# df_sku_seller.columns


# In[8]:


clear_seller = df_seller.rename(columns={'_id':'seller_id'})


# In[9]:


clear_sku_seller = pd.merge(df_sku_seller,clear_seller, on='seller_id', how='inner')
clear_sku_seller.drop(columns=['height_nu','depth_nu','width_nu','status_nm','created_at_dt_y','updated_at_dt_y'],inplace=True)
clear_sku_seller.drop(columns=['documents','contacts','image','stores','logo_url_ds'],inplace=True)
clear_sku_seller.columns


# In[10]:


ind1 = clear_sku_seller[(clear_sku_seller['status_ds'] != 'ACTIVE')].index
clear_sku_seller.drop(index=ind1,inplace=True)
ind2 = clear_sku_seller[clear_sku_seller['product_id'].isnull()].index
clear_sku_seller.drop(index=ind2,inplace=True)
ind3 = clear_sku_seller[clear_sku_seller['seller_type_ds'] == 'INTERNAL'].index
clear_sku_seller.drop(index=ind3,inplace=True)
clear_sku_seller.drop(columns=['status_ds','seller_type_ds'],inplace=True)


# In[11]:


clear_sku_seller.shape # 53 segmentos únicos


# In[12]:


# clear_sku_seller.head()


# In[13]:


import numpy as np
corpus = pd.DataFrame(data=clear_sku_seller[{'_id','sku_ds','ean_cd','segment_id','segment_nm'}].iloc[0:500],columns={'_id','sku_ds','ean_cd','segment_id','segment_nm'})
corpus.set_index('_id')
corpus['qtd'] = np.ones(500)
print(corpus.shape) # documents df apenas com id, codigo sku seller e descrição
print(corpus['segment_nm'].unique())
print(len(corpus['segment_nm'].unique()))


# In[14]:


import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[15]:


from bs4 import BeautifulSoup
import re
corpus['sku_ds'].dropna(inplace=True)
doc_html = []
for doc in corpus['sku_ds']:
    doc_html.append(BeautifulSoup(doc, "lxml").text)


# In[16]:


doc_html[1:2]


# In[56]:


segmentos = corpus['segment_nm']


# In[18]:


# função de remoção de tags html
def remove_html_tags(text):
    # Remove html tags de um texto qualquer
    clean = re.compile('<.*?>')
    return re.sub(clean, '', text)


# In[ ]:





# In[19]:


import unicodedata
# normalização do texto e remoção das tags
doc_no_tags = []
for doc in doc_html:
    doc_no_tags.append(remove_html_tags(unicodedata.normalize("NFKC",doc)))


# In[20]:


# lower case
doc_no_tags = [x.lower() for x in doc_no_tags]
doc_no_tags[1:2]


# In[21]:


# remoção de números
doc_no_numbers = []
for doc in doc_no_tags:
    doc_no_numbers.append(re.sub(r'\d+', '', doc))
doc_no_numbers


# In[22]:


doc_no_numbers[1:2]


# In[23]:


# remoção de pontuação
from nltk.tokenize import RegexpTokenizer
doc_no_punctuation = []
tokenizer = RegexpTokenizer(r'\w+')
for doc in doc_no_numbers:
    doc_no_punctuation.append(doc)
doc_no_punctuation[1:2]


# In[24]:


# remoção de stopwords
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
doc_no_stopwords = []
stopWords = set(stopwords.words('portuguese'))
for doc in doc_no_punctuation:
    wordsFiltered = []
    for palavra in doc.split():
        if palavra not in stopWords:
            wordsFiltered.append(palavra)
    doc_no_stopwords.append(' '.join(wordsFiltered))
doc_no_stopwords[1:2]


# In[25]:


# lemmatization
import spacy
nlp_pt = spacy.load('pt_core_news_sm')
doc_lem = []
for doc in doc_no_stopwords:
    doc = nlp_pt(doc)
    doc_lem.append(' '.join([token.lemma_ for token in doc]))
doc_lem[1:2]


# In[26]:


from nltk.stem import PorterStemmer
from nltk.stem import LancasterStemmer


# In[27]:


porter = PorterStemmer()
lancaster=LancasterStemmer()


# In[28]:


# stemming Porter
doc_porter_stem = []
for doc in doc_no_stopwords:
    doc_porter_stem.append([porter.stem(token) for token in doc.split()])


# In[29]:


# stemming Lancaster
doc_lancaster_stem = []
for doc in doc_no_stopwords:
    doc_lancaster_stem.append([lancaster.stem(token) for token in doc.split()])


# In[ ]:





# In[30]:


# counting - frequencia
from sklearn.feature_extraction.text import CountVectorizer
vec = CountVectorizer() # we could ignore binary=False argument since it is default
vec.fit(doc_lem)

import pandas as pd
pd.DataFrame(vec.transform(doc_lem).toarray(), columns=sorted(vec.vocabulary_.keys()))


# In[31]:


# TF-IDF
from sklearn.feature_extraction.text import TfidfVectorizer
vec = TfidfVectorizer()
vec.fit(doc_lem)

import pandas as pd
features = pd.DataFrame(vec.transform(doc_lem).toarray(), columns=sorted(vec.vocabulary_.keys()))


# In[32]:


# treinamento do modelo com k-means
from sklearn.cluster import MiniBatchKMeans
cls = MiniBatchKMeans(n_clusters=18)
cls.fit(features)


# In[33]:


# predict cluster labels for new dataset
cls.predict(features)

# to get cluster labels for the dataset used while
# training the model (used for models that does not
# support prediction on new dataset).
print(cls.labels_)
print(len(cls.labels_))


# In[77]:


indices = pd.DataFrame(segmentos.unique())
segmentos.to_list()
for pos, elem in enumerate()


# In[34]:


from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
# visualization
# reduce the features to 2D
pca = PCA(n_components=3)
reduced_features = pca.fit_transform(features)

# reduce the cluster centers to 2D
reduced_cluster_centers = pca.transform(cls.cluster_centers_)


# In[35]:


plt.scatter(reduced_features[:,0], reduced_features[:,1], c=cls.predict(features))
plt.scatter(reduced_cluster_centers[:, 0], reduced_cluster_centers[:,1], marker='x', s=150, c='b')


# In[36]:


from bs4 import BeautifulSoup
import re

# função de remoção de html tags por texto
def remove_html_tags(text):
    clean = re.compile('<.*?>')
    return re.sub(clean, '', text)

# normalização do texto e remoção das tags
for doc in corpus:
    doc_no_tags.append(remove_html_tags(unicodedata.normalize("NFKC",BeautifulSoup(doc, "lxml").text)))    


# In[ ]:





# In[37]:


from spellchecker import SpellChecker

spell = SpellChecker(language='pt')

# find those words that may be misspelled
misspelled = spell.unknown(['baton','com','acababamento'])

for word in misspelled:
    # Get the one `most likely` answer
    print(spell.correction(word))

    # Get a list of `likely` options
    print(spell.candidates(word))


# In[ ]:





# In[38]:


nlp_pt.to_disk("./model") 


# In[ ]:




