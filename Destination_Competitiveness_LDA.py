#!/usr/bin/env python
# coding: utf-8

# In[104]:


import re
import numpy as np
import pandas as pd
from pprint import pprint


# Gensim
import gensim
import gensim.corpora as corpora
from gensim.utils import simple_preprocess
from gensim.models import CoherenceModel

# spacy for lemmatization
import spacy

# Plotting tools
import pyLDAvis
#import pyLDAvis.gensim  # don't skip this
import matplotlib.pyplot as plt
#matplotlib inline

# Enable logging for gensim - optional
import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.ERROR)

import warnings
warnings.filterwarnings("ignore",category=DeprecationWarning)


# NLTK Stop words
from nltk.corpus import stopwords
stop_words = stopwords.words('english')
stop_words.extend(['paper', 'abstract', 'literature', 'review', 'case', 'study', 'result', 'reference', 'keyword', 'doi', 'et_al', 'international',
                  'tourism', 'destinations', 'competitiveness', 'journal', 'yangon', 'nanyang', 'adb', 'ghosh', 'ghimere', 'funatsu', 'inle',
                  'cbdbe', 'Shoval', 'imc', 'birenboim', 'schultz', 'saunder', 'research', 'also','destination', 'management', 'tourist'])


# In[123]:


from os import listdir
import codecs


literaturePath = "E:/PHD/Paper/Destination Competitiveness Review/Result/data/"
literatures = [literaturePath + f for f in listdir(literaturePath)]

def read_data(dataFile):
    reader = codecs.open(dataFile, mode="r", encoding="utf-8")
    content = ""
    for line in reader.readlines():
        content = content + line
    return content

def readLiteratureCotent(literatures):
    for path in literatures:
        yield(read_data(path))
        
        
data = list(readLiteratureCotent(literatures))


# Remove Emails
data = [re.sub('\S*@\S*\s?', '', sent) for sent in data]


# Remove new line characters
data = [re.sub('\s+', ' ', sent) for sent in data]

# Remove distracting single quotes
data = [re.sub("\'"," ", sent) for sent in data]

# Remove distracting single quotes
data = [re.sub("\n\n\x0c", "", sent) for sent in data]
data = [re.sub("ª", "", sent) for sent in data]

print(data[:1])    


# In[167]:


def sent_to_words(sentences):
    for sentence in sentences:
        yield(gensim.utils.simple_preprocess(str(sentence).encode('utf-8'), deacc=True))  # deacc=True removes punctuations

data_words = list(sent_to_words(data))
print(data_words[:1])


# In[208]:


# Build the bigram and trigram models
bigram = gensim.models.Phrases(data_words, min_count = 5, threshold = 150) # higher threshold fewer phrases.
# trigram = gensim.models.Phrases(bigram[data_words], threshold = 100)  

# Faster way to get a sentence clubbed as a trigram/bigram
bigram_mod = gensim.models.phrases.Phraser(bigram)
# trigram_mod = gensim.models.phrases.Phraser(trigram)

# See trigram example
print(bigram_mod[data_words[0]])


# In[209]:


# Define functions for stopwords, bigrams, trigrams and lemmatization
def remove_stopwords(texts):
    return [[word for word in simple_preprocess(str(doc)) if word not in stop_words] for doc in texts]

def make_bigrams(texts):
    return [bigram_mod[doc] for doc in texts]

def make_trigrams(texts):
    return [trigram_mod[bigram_mod[doc]] for doc in texts]

def lemmatization(texts, allowed_postags=['NOUN', 'ADJ', 'ADV']):
    """https://spacy.io/api/annotation"""
    texts_out = []
    for sent in texts:
        doc = nlp(" ".join(sent)) 
        texts_out.append([token.lemma_ for token in doc if token.pos_ in allowed_postags])
    return texts_out

# Remove Stop Words
data_words_nostops = remove_stopwords(data_words)

# Form Bigrams
data_words_bigrams = make_bigrams(data_words_nostops)

# Initialize spacy 'en' model, keeping only tagger component (for efficiency)
# python3 -m spacy download en
nlp = spacy.load('en_core_web_sm', disable=['parser', 'ner'])

# Do lemmatization keeping only noun, adj, vb, adv
data_lemmatized = lemmatization(data_words_bigrams, allowed_postags=['NOUN', 'ADJ', 'ADV'])

print(data_lemmatized[:1])


# In[198]:


# Create Dictionary
id2word = corpora.Dictionary(data_lemmatized)

# Create Corpus
texts = data_lemmatized

# Term Document Frequency
corpus = [id2word.doc2bow(text) for text in texts]

def compute_coherence_values(dictionary, corpus, texts, limit, start=2, step=2):
    """
    Compute c_v coherence for various number of topics

    Parameters:
    ----------
    dictionary : Gensim dictionary
    corpus : Gensim corpus
    texts : List of input texts
    limit : Max num of topics

    Returns:
    -------
    model_list : List of LDA topic models
    coherence_values : Coherence values corresponding to the LDA model with respective number of topics
    """
    coherence_values = []
    model_list = []
    for num_topics in range(start, limit, step):
#         model = gensim.models.wrappers.LdaMallet(mallet_path, corpus=corpus, num_topics=num_topics, id2word=id2word)
        model = gensim.models.ldamodel.LdaModel(corpus=corpus,
                                           id2word=id2word,
                                           num_topics=num_topics, 
                                           random_state=100,
                                           update_every=1,
                                           chunksize=20,
                                           passes=2,
                                           alpha='auto',
                                           per_word_topics=True)
        model_list.append(model)
        coherencemodel = CoherenceModel(model=model, texts=texts, dictionary=dictionary, coherence='c_v')
        coherence_values.append(coherencemodel.get_coherence())

    return model_list, coherence_values


# In[210]:


# Can take a long time to run.
model_list, coherence_values = compute_coherence_values(dictionary=id2word, corpus=corpus, texts=data_lemmatized, start=4, limit=15, step=1)


# In[211]:


# Show graph
limit=15; start=4; step=1;
x = range(start, limit, step)
plt.plot(x, coherence_values)
plt.xlabel("Num Topics")
plt.ylabel("Coherence score")
plt.legend(("coherence_values"), loc='best')
plt.show()


# In[212]:


# Print the coherence scores
for m, cv in zip(x, coherence_values):
    print("Num Topics =", m, " has Coherence Value of", round(cv, 4))


# In[195]:


# Select the model and print the topics
optimal_model = model_list[6]
model_topics = optimal_model.show_topics(formatted=False)
pprint(optimal_model.print_topics(num_words=10))


# In[213]:


# Visualize the topics
from pyLDAvis import gensim_models
pyLDAvis.enable_notebook()
vis = gensim_models.prepare(optimal_model, corpus, id2word)
vis


# In[221]:


from wordcloud import (WordCloud, get_single_color_func)
import matplotlib.pyplot as plt


class SimpleGroupedColorFunc(object):
    """Create a color function object which assigns EXACT colors
       to certain words based on the color to words mapping

       Parameters
       ----------
       color_to_words : dict(str -> list(str))
         A dictionary that maps a color to the list of words.

       default_color : str
         Color that will be assigned to a word that's not a member
         of any value from color_to_words.
    """

    def __init__(self, color_to_words, default_color):
        self.word_to_color = {word: color
                              for (color, words) in color_to_words.items()
                              for word in words}

        self.default_color = default_color

    def __call__(self, word, **kwargs):
        return self.word_to_color.get(word, self.default_color)


class GroupedColorFunc(object):
    """Create a color function object which assigns DIFFERENT SHADES of
       specified colors to certain words based on the color to words mapping.

       Uses wordcloud.get_single_color_func

       Parameters
       ----------
       color_to_words : dict(str -> list(str))
         A dictionary that maps a color to the list of words.

       default_color : str
         Color that will be assigned to a word that's not a member
         of any value from color_to_words.
    """

    def __init__(self, color_to_words, default_color):
        self.color_func_to_words = [
            (get_single_color_func(color), set(words))
            for (color, words) in color_to_words.items()]

        self.default_color_func = get_single_color_func(default_color)

    def get_color_func(self, word):
        """Returns a single_color_func associated with the word"""
        try:
            color_func = next(
                color_func for (color_func, words) in self.color_func_to_words
                if word in words)
        except StopIteration:
            color_func = self.default_color_func

        return color_func

    def __call__(self, word, **kwargs):
        return self.get_color_func(word)(word, **kwargs)


# In[305]:


from wordcloud import WordCloud

T1 = {"resource":0.014,"model":0.009,"event":0.008,"strategy":0.008,"marketing":0.007,"dmo":0.007,"tourist":0.006,"strategic":0.006,"service":0.006,"competitive":0.006}
T2 = {"factor":  0.010,"model":0.009,"country":0.007,"travel":0.007,"resource":0.007,"tourist":0.007,"analysis": 0.007,"economic":0.007,"indicator":0.006,"development":0.006}
T3 = {"shopping":0.013,"online_reputation":0.007,"store":0.005,"bda":0.005,"tourist":0.005,"product":0.004,"orm":0.003,"service":0.003,"marketing":0.003,"korea":0.003}
T4 = {"bali":0.005,"taxi":0.004,"tnc":0.004,"cluster_actor":0.003,"electre":0.003,"wick":0.003,"development":0.002,"cluster":0.002,"balinese":0.002,"interview":0.002}
T5 = {"price":0.017,"country":0.017,"index":0.013,"variable":0.012,"model":0.009,"analysis":0.009,"datum":0.007,"economic":0.007,"policy":0.006,"spain":0.006}
T6 = {"innovation":0.013,"cluster":0.009,"ﬁrm":0.008,"rural":0.008,"company":0.007,"development":0.007,"knowledge":0.006,"service":0.006,"network":0.006,"cid":0.005}
T7 = {"model":0.001,"tourist":0.001,"analysis":0.001,"resource":0.001,"factor":0.000,"competitive":0.000,"shopping":0.000,"marketing":0.000,"development":0.000,"business":0.000}
T8 = {"hotel":0.017,"efficiency":0.017,"city":0.016,"analysis":0.012,"number":0.010,"dea":0.009,"total":0.009,"economic":0.008,"mediation":0.008,"model":0.008}
T9 = {"russia":0.017,"russian":0.008,"transit":0.002,"andrades_dimanche":0.002,"tourist":0.001,"job":0.001,"development":0.001,"sheresheva":0.001,"travel":0.001,"competitive":0.001}
T10 = {"transit":0.003,"job":0.001,"tourist":0.001,"experience":0.001,"region":0.001,"access":0.001,"travel":0.001,"accessibility":0.001,"factor":0.001,"competitive":0.001}
# Generate a word cloud image

fig, axs = plt.subplots(nrows=4,ncols=3,figsize=(15,15))
default_color = 'slategray'


def addSubPlot(Ti, index, title):
    keyws = list(Ti.keys())
    color_to_words = {
        'fuchsia': [keyws[0]],
        'plum': keyws[:3]
    }
    wordcloud = WordCloud(color_func=GroupedColorFunc(color_to_words, default_color), background_color="rgba(255, 255, 255, 0)", mode="RGBA").generate_from_frequencies(Ti)
    axs = fig.add_subplot(4,3,index)
    axs.imshow(wordcloud)
    axs.set_title(title)
    axs.axis("off")

addSubPlot(T1, 1, 'T1')  
addSubPlot(T2, 2, 'T2') 
addSubPlot(T3, 3, 'T3')
addSubPlot(T4, 4, 'T4')
addSubPlot(T5, 5, 'T5')
addSubPlot(T6, 6, 'T6')
addSubPlot(T7, 7, 'T7')
addSubPlot(T8, 8, 'T8')
addSubPlot(T9, 9, 'T9')
addSubPlot(T10, 11, 'T10')

axs[-1, -1].axis('off')
axs[-1, -3].axis('off')






