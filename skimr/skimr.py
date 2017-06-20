#!/usr/bin/env python

###################################################################################################
# SKIMR for the web:
#
# -Use selenium webdriver to scrape html from input url
# -Apply cleaning function to html
# -Apply model to sentences
# -Find sentences in article
# -Apply markup to sentences
# -Show article with markups
#
# Clarence Cheng, 2017



# WEB
from flask import render_template, request, redirect, url_for
from skimr import app
from selenium import webdriver
from selenium.webdriver.common.keys import Keys
from bs4 import BeautifulSoup

# TOOLS
import re, os, time, sys
import pandas as pd
import numpy as np
import scipy
from scipy import spatial
import pickle
import math
import string
from string import whitespace, punctuation

# NLP
sys.path.insert(0, 'readability')   # readability from https://github.com/mmautner/readability
from readability import Readability
import nltk.data
sent_tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
from nltk.tokenize import RegexpTokenizer
word_tokenizer = RegexpTokenizer('\s+', gaps=True)
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
p_stemmer = PorterStemmer()

from patsy import dmatrices

from stop_words import get_stop_words
stop_en     = get_stop_words('en')
en_words    = set(nltk.corpus.words.words())
stopw_en    = stopwords.words('english')
all_stopw   = set(stopw_en) | set(stop_en)

pipe            = pd.read_pickle('pkl/model_logreg.pkl')
ldamodel        = pickle.load(open('pkl/lda_model.pkl','rb'))
commonwords_2   = pickle.load(open('pkl/commonwords.pkl','rb'))
dictionary      = pickle.load(open('pkl/lda_dictionary.pkl','rb'))

#############################################################################################################################
def webscrape(inputurl):
    # scrape html from url
    drvr = webdriver.PhantomJS()
    drvr.get(inputurl)
    html1 = drvr.page_source
    html2 = drvr.execute_script('return document.documentElement.innerHTML;')
    return html2
#############################################################################################################################

#############################################################################################################################
def cleanfulltext(scrapedhtml):

    lines = []
    soup  = BeautifulSoup(scrapedhtml,'lxml')                     
    txt0  = soup.find('div',attrs={'data-source':'post_page'})    
    txt1  = txt0.find_all(class_='graf')

    for line in txt1:
        txt2 = re.sub('<[^>]+>', '', str(line) )
        lines.append(txt2)

    fulltext = ' '.join(lines)

    return fulltext
#############################################################################################################################

#############################################################################################################################
def calc_params(fulltext):
# Calculate values for logistic regression features

# tokenize full text into sentences
# for each sentence, calculate:
#   sentence length
#   readability metrics
#   LDA vector of sentence -> cos similarity to LDA vector of article
#   article position
# put in array:
#   id     sentence length     grade level     LDA similarity


    all_sents = []
    all_ARI = []
    all_FRE = []
    all_FKG = []
    all_GFI = []
    all_SMG = []
    all_CLI = []
    all_LIX = []
    all_RIX = []
    all_lens = []
    all_ldadists = []
    all_wposes = []
    all_sposes = []

    # Get topic vector for the whole article
    fulltext_prep = prep_text(fulltext)
    # convert article to corpus (bag-of-words)
    text_corpus = dictionary.doc2bow(fulltext_prep)
    # calculate document lda
    doc_lda = ldamodel[text_corpus]
    # convert lda to vector
    vec_lda_art = lda_to_vec(doc_lda)

    # count lengths of non-highlighted sentences
    ftsents = sent_tokenizer.tokenize(fulltext)
    for f in ftsents:
        # get LDA metric
        f_clean = prep_text(f)
        f_corpus = dictionary.doc2bow(f_clean)
        sent_lda = ldamodel[f_corpus]
        vec_lda = lda_to_vec(sent_lda)
        f_lda = 1-spatial.distance.cosine(vec_lda, vec_lda_art)
        np_vec_lda = np.asarray(vec_lda)
        all_ldadists.append(f_lda)

        # get fraction position
        f_wpos, f_spos = sent_pos(f, fulltext)#f[:-2], fulltext)
        all_wposes.append( float(f_wpos) )
        all_sposes.append( float(f_spos) )
        
        # get length
        ftwords = word_tokenizer.tokenize(f)
        ftlen = len(ftwords)
        all_lens.append(int(ftlen))

        # get readability
        f_rd = Readability(f)
        all_ARI.append( float(f_rd.ARI()) )
        all_FRE.append( float(f_rd.FleschReadingEase()) )
        all_FKG.append( float(f_rd.FleschKincaidGradeLevel()) )
        all_GFI.append( float(f_rd.GunningFogIndex()) )
        all_SMG.append( float(f_rd.SMOGIndex()) )
        all_CLI.append( float(f_rd.ColemanLiauIndex()) )
        all_LIX.append( float(f_rd.LIX()) )
        all_RIX.append( float(f_rd.RIX()) )

        # sentence
        all_sents.append(f)


    # build pandas dataframe
    data = pd.DataFrame({ \
                          'dummy':all_lens, \
                          'sentences':all_sents, \
                          'length':all_lens, \
                          'LDAdist':all_ldadists, \
                          'wordPos':all_wposes, \
                          'sentPos':all_sposes, \
                          'ARI':all_ARI, \
                          'FRE':all_FRE, \
                          'FKG':all_FKG, \
                          'GFI':all_GFI, \
                          'SMG':all_SMG, \
                          'CLI':all_CLI, \
                          'LIX':all_LIX, \
                          'RIX':all_RIX, \
                        })
    
    return data    #pandas dataframe with all sentences and their feature values
#############################################################################################################################

#############################################################################################################################
def prep_text(sent):   # for cleaning individual sentences
    # remove punctuation
    translator = str.maketrans('', '', string.punctuation)
    txt2 = re.sub(u'\u2014','',sent) # remove em dashes
    txt3 = re.sub(r'\d+', '', txt2) # remove digits
    txt4 = txt3.translate(translator) # remove punctuation
    # split text into words
    tokens = word_tokenizer.tokenize(txt4.lower())
    # strip single and double quotes from ends of words
    tokens_strip = [i.strip('”“’‘') for i in tokens]
    # keep only english words
    tokens_en = [i for i in tokens_strip if i in en_words]
    # remove nltk/stop_word stop words
    nostop_tokens = [i for i in tokens_en if not i in all_stopw]
    # strip single and double quotes from ends of words
    nostop_strip = [i.strip('”“’‘') for i in nostop_tokens]
    # stem words
    stemmed = [p_stemmer.stem(i) for i in nostop_strip]
    # strip single and double quotes from ends of words
    stemmed_strip = [i.strip('”“’‘') for i in stemmed]
    # stem words
    stemmed2 = [p_stemmer.stem(i) for i in stemmed_strip]
    # strip single and double quotes from ends of words
    stemmed2_strip = [i.strip('”“’‘') for i in stemmed2]
    # remove common words post-stemming
    stemmed_nocommon = [i for i in stemmed2_strip if not i in commonwords_2]
    return stemmed_nocommon
#############################################################################################################################

#############################################################################################################################
def lda_to_vec(lda_input):
    num_topics = 10
    vec = [0]*num_topics
    for i in lda_input:
        col = i[0]
        val = i[1]
        vec[col] = val
    return vec
#############################################################################################################################

#############################################################################################################################
# Function to calculate position of sentence within article
# (frac of sentences into text)

def sent_pos(sentence, text):

    # sentence is tokenized from highlight or full text
    
    # remove 1-word sentences?
    
    # break text into sentences and get total sents in full text
    full_sents = sent_tokenizer.tokenize(text)
    num_sents = len(full_sents)

    # break text into words and get total words in full text
    full_words = word_tokenizer.tokenize(text)
    num_words = len(full_words)

    pos = text.find(sentence)

    if pos >= 0:

        # total words in full text before highlight position
        b4_words = word_tokenizer.tokenize(text[:pos])
        b4_wlen = len(b4_words)

        # sentences in full text before highlight position
        b4_sents = sent_tokenizer.tokenize(text[:pos])
        b4_slen = len(b4_sents)

        frc_w = b4_wlen / num_words
        frc_s = b4_slen / num_sents

    elif pos < 0:
        frc_w = -1
        frc_s = -1
                
    return frc_w, frc_s

#############################################################################################################################

#############################################################################################################################
def predict(data):
    y, X = dmatrices('dummy ~ length + LDAdist + wordPos + sentPos + ARI + FRE + FKG + GFI + SMG + CLI + LIX + RIX', \
                 data, return_type="dataframe")

    y = np.ravel(y)
    # predict value for data
    predicted = pipe.predict(X)
    # get confidence score
    decfxn = pipe.decision_function(X)
    test = np.nonzero(predicted)
    return predicted, decfxn
#############################################################################################################################

#############################################################################################################################
def markup(predicted, decfxn, data, scrapedhtml):

    soup = BeautifulSoup(scrapedhtml,'lxml')

    # USE CONFIDENCE SCORES TO ONLY HIGHLIGHT 30% OF SENTENCES?
    # predict = list(predicted)
    # num_sents = len(predict)
    # highlight_thres = 0.3
    # num_highlight = int(math.floor(0.3*num_sents))
    # decfxn = pipe.decision_function(X)
    # cumul = pd.DataFrame({'confidencescore':decfxn, 'highlightornot':predict})
    # cumul_sort = cumul.sort_values('confidencescore', ascending=False)
    # predict_thres = []
    # for i in cumul_sort['highlightornot']:
    #     if total <= num_highlight:
    #         total = total + i
    #         predict_thres.append(i)
    #     elif total > num_highlight:
    #         predict_thres.append(0)


    predict = list(predicted)
    tmpsoup = str(soup)
    decision = list(decfxn)
    n = 0
    for f in data['sentences']:
        # time.sleep(10)
        # newsoup = soup
        # print(n)
        # print(predict[n])
        # print(f)
        # print(decision[n])
        # if n <= 5:
        #     print(soup)
        if predict[n] == 1:
            if decision[n] >= 0.1:
                # print(str(f)+' is highlighted')
                newf = '<span style="background-color: #4EE2EC">'+f+'</span>'
                # print('newf is: ' + newf)
                # if n <= 5:
                #     print(soup)
                # print(soup)
                tmpsoup = tmpsoup.replace( f, newf, 1)
                # print(tmpsoup)

        n+=1

    outsoup = BeautifulSoup(tmpsoup, 'lxml')
    htmlmarkup = outsoup.prettify()
    return htmlmarkup
#############################################################################################################################




@app.route('/')

@app.route('/index')
def index():
    return render_template("index.html")

@app.route('/output')
def output():
    inputtext = request.args.get('inputtext')
    scrapedhtml = webscrape(inputtext)
    cleanedtext = cleanfulltext(scrapedhtml)
    data = calc_params(cleanedtext)
    predicted, decfxn = predict(data)
    htmlmarkup = markup(predicted, decfxn, data, scrapedhtml)
    return render_template("output.html", html=htmlmarkup)





