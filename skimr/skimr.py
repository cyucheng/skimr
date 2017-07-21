#!/usr/bin/env python


"""
skimr is a web application for streamlining reading of articles online.
It currently works for articles on Medium.com but could be adapted for articles
on other websites with minor adjustments to the html cleaning function.

The framework of skimr is:
- Use selenium webdriver to scrape full HTML of article from user-input url
- Apply a cleaning function to HTML to get full text of the article
- Calculate feature values for sentences
  - Features include topic similarity score between sentence and the article
    it's from, sentence length, sentence position, and readability metrics
  - Topic distributions calculated using Latent Dirichlet Allocation (LDA)
- Use pre-trained logistic regression model to predict highlighted sentences
- Find sentences to be highlighted in HTML of article
- Apply markup to sentences in HTML
- Display article with markups in the browser

To see the full skimr package, visit https://github.com/cyucheng/skimr

Clarence Cheng, 2017
"""


###############################################################################
# Imports

# WEB
from flask import render_template, request
from skimr import app
from selenium import webdriver
from bs4 import BeautifulSoup

# TOOLS
import re
import sys
import pandas as pd
import numpy as np
from scipy import spatial
import pickle
import string

# ML/NLP
from patsy import dmatrices
sys.path.insert(0, 'readability')       # From mmautner/readability on GitHub
from readability import Readability                 # noqa
import nltk                                         # noqa
import nltk.data                                    # noqa
from nltk.tokenize import RegexpTokenizer           # noqa
from nltk.corpus import stopwords                   # noqa
from nltk.stem.porter import PorterStemmer          # noqa
from stop_words import get_stop_words               # noqa

# Set up tokenizers/stemmers/stopword lists
sent_tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
word_tokenizer = RegexpTokenizer('\s+', gaps=True)
p_stemmer = PorterStemmer()
stop_en = get_stop_words('en')
en_words = set(nltk.corpus.words.words())
stopw_en = stopwords.words('english')
all_stopw = set(stopw_en) | set(stop_en)

# Load pickled files
pipe = pd.read_pickle('pkl/model_logreg_std.pkl')  # Logistic regression model
ldamodel = pickle.load(open('pkl/lda_model.pkl', 'rb'))  # LDA model
commonwords_2 = pickle.load(open('pkl/commonwords.pkl', 'rb'))  # Common words
dictionary = pickle.load(open('pkl/lda_dictionary.pkl', 'rb'))  # LDA dictionary  # noqa


###############################################################################
# Define functions

def webscrape(inputurl):
    """
    Retrieves the HTML source for a given URL.

    Args:
    - inputurl (str): URL for a webpage

    Returns:
    - html (str): full HTML from webpage
    """

    drvr = webdriver.PhantomJS()
    drvr.get(inputurl)
    html = drvr.execute_script('return document.documentElement.innerHTML;')

    return html
###############################################################################


def getfulltext(scrapedhtml):
    """
    Gets the full text of the article from the HTML source.

    Args:
    - scrapedhtml (str): full HTML from webpage

    Returns:
    - fulltext (str): full text of article from webpage
    """

    # Get text from paragraphs inside tag for body of article
    lines = []
    soup = BeautifulSoup(scrapedhtml, 'lxml')
    txt0 = soup.find('div', attrs={'data-source': 'post_page'})
    txt1 = txt0.find_all(class_='graf')

    # Remove HTML tags
    for line in txt1:
        txt2 = re.sub('<[^>]+>', '', str(line))
        lines.append(txt2)

    # Join into full text string
    fulltext = ' '.join(lines)

    return fulltext
###############################################################################


def calc_params(fulltext):
    """
    Calculate feature values for each sentence in the article.

    Args:
    - fulltext (str): full text of article from webpage

    Returns:
    - data (df): dataframe with feature values for each sentence

    Steps:
    - Tokenize full text into sentences
    - For each sentence, calculate:
      - Topic similarity score: cosine similarity of sentence topic
        distribution to article topic distribution
      - Position within article in fraction of words and sentences (0 = start
        of article, 1 = end of article)
      - Sentence length
      - Readability metrics
    - Put feature values in dataframe
    """

    # Initialize lists for each feature; will be inputs to dataframe
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

    # Compute topic vector for the whole article
    #   Clean full text
    fulltext_prep = clean_text(fulltext)
    #   Convert article to bag-of-words
    text_corpus = dictionary.doc2bow(fulltext_prep)
    #   Calculate document topic distribution
    doc_lda = ldamodel[text_corpus]
    #   Convert topic distribution to vector
    vec_lda_art = lda_to_vec(doc_lda)

    # Break full text into sentences
    ftsents = sent_tokenizer.tokenize(fulltext)

    for f in ftsents:

        # Get topic similarity score of sentence vs article
        f_clean = clean_text(f)
        f_corpus = dictionary.doc2bow(f_clean)
        sent_lda = ldamodel[f_corpus]
        vec_lda = lda_to_vec(sent_lda)
        f_lda = 1-spatial.distance.cosine(vec_lda, vec_lda_art)
        all_ldadists.append(f_lda)

        # Get sentence position (fraction way through article)
        f_wpos, f_spos = sent_pos(f, fulltext)
        all_wposes.append(float(f_wpos))
        all_sposes.append(float(f_spos))

        # Get length of sentence
        ftwords = word_tokenizer.tokenize(f)
        ftlen = len(ftwords)
        all_lens.append(int(ftlen))

        # Get readability metrics
        f_rd = Readability(f)
        all_ARI.append(float(f_rd.ARI()))
        all_FRE.append(float(f_rd.FleschReadingEase()))
        all_FKG.append(float(f_rd.FleschKincaidGradeLevel()))
        all_GFI.append(float(f_rd.GunningFogIndex()))
        all_SMG.append(float(f_rd.SMOGIndex()))
        all_CLI.append(float(f_rd.ColemanLiauIndex()))
        all_LIX.append(float(f_rd.LIX()))
        all_RIX.append(float(f_rd.RIX()))

        # sentence
        all_sents.append(f)

    # Build pandas dataframe
    data = pd.DataFrame({
                          'dummy': all_lens,
                          'sentences': all_sents,
                          'length': all_lens,
                          'LDAdist': all_ldadists,
                          'wordPos': all_wposes,
                          'sentPos': all_sposes,
                          'ARI': all_ARI,
                          'FRE': all_FRE,
                          'FKG': all_FKG,
                          'GFI': all_GFI,
                          'SMG': all_SMG,
                          'CLI': all_CLI,
                          'LIX': all_LIX,
                          'RIX': all_RIX,
                        })

    return data
###############################################################################


def clean_text(text):
    """
    Clean text of full article or individual sentences so features can be
    calculated.

    Args:
    - text (str): full text of article or individual sentence

    Returns:
    - stemmed_nocommon (list): list of processed words in text

    Steps:
    - Remove punctuation
    - Split text into words
    - Strip single and double quotes from ends of words
    - Remove non-English words
    - Remove stopwords
    - Ensure no quotes in words before stemming
    - Stem words
    - Remove any quotes remaining after stemming
    - Stem words again to account for words 'masked' by quotes
    - Final pass to remove any remaining quotes
    - Remove common words, post-stemming
      - Common words are those appearing in >=60% of documents (calculated
        separately in 4_LDA_analysis.ipynb in skimr/jupyter on GitHub)
    """

    translator = str.maketrans('', '', string.punctuation)
    txt2 = re.sub(u'\u2014', '', text)  # Remove em dashes
    txt3 = re.sub(r'\d+', '', txt2)     # Remove digits
    txt4 = txt3.translate(translator)   # Remove punctuation

    tokens = word_tokenizer.tokenize(txt4.lower())
    tokens_strip = [i.strip('”“’‘') for i in tokens]
    tokens_en = [i for i in tokens_strip if i in en_words]
    nostop_tokens = [i for i in tokens_en if not (i in all_stopw)]
    nostop_strip = [i.strip('”“’‘') for i in nostop_tokens]
    stemmed = [p_stemmer.stem(i) for i in nostop_strip]
    stemmed_strip = [i.strip('”“’‘') for i in stemmed]
    stemmed2 = [p_stemmer.stem(i) for i in stemmed_strip]
    stemmed2_strip = [i.strip('”“’‘') for i in stemmed2]
    stemmed_nocommon = [i for i in stemmed2_strip if not (i in commonwords_2)]

    return stemmed_nocommon
###############################################################################


def lda_to_vec(lda_input):
    """
    Convert topic distribution from LDA to a numeric vector that can be
    compared to others.

    Args:
    - lda_input (list): list of tuples [topic_id, topic_probability] output by
                        LDA model

    Returns:
    - vec (list): list of topic probabilities
    """

    num_topics = 10
    vec = [0]*num_topics

    for i in lda_input:
        col = i[0]
        val = i[1]
        vec[col] = val

    return vec
###############################################################################


def sent_pos(sentence, text):
    """
    Calculate position of sentence in article as the fraction of words and
    sentences into the text.

    Args:
    - sentence (str): sentence for which to calculate this
    - text (str): full text of article

    Returns:
    - frc_w (float): fraction of words into the text that sentence begins
    - frc_s (float): fraction of sentences into the text that sentence begins
    """

    # Break text into sentences and get total sents in full text
    full_sents = sent_tokenizer.tokenize(text)
    num_sents = len(full_sents)

    # Break text into words and get total words in full text
    full_words = word_tokenizer.tokenize(text)
    num_words = len(full_words)

    pos = text.find(sentence)

    if pos >= 0:

        # Total words in full text before highlight position
        b4_words = word_tokenizer.tokenize(text[:pos])
        b4_wlen = len(b4_words)

        # Sentences in full text before highlight position
        b4_sents = sent_tokenizer.tokenize(text[:pos])
        b4_slen = len(b4_sents)

        frc_w = b4_wlen / num_words
        frc_s = b4_slen / num_sents

    elif pos < 0:

        # If sentence not found in article, set fraction to -1 (there may be a
        #   better way to do this, e.g. make a categorical variable for missing
        #   position?)
        frc_w = -1
        frc_s = -1

    return frc_w, frc_s

###############################################################################


def predict(data):
    """
    Predict category (0 = non-highlighted, 1 = highlighted) and confidence
    score for each sentence.

    Args:
    - data (df): dataframe with feature values for each sentence

    Returns:
    - predicted (array): predicted category for each sentence
    - decfxn (array): confidence score for each sentence
    """

    y, X = dmatrices('dummy ~ length + LDAdist + wordPos + sentPos + ARI + FRE \
                     + FKG + GFI + SMG + CLI + LIX + RIX',
                     data, return_type="dataframe")

    y = np.ravel(y)

    # Predict value for data
    predicted = pipe.predict(X)
    # Get confidence score
    decfxn = pipe.decision_function(X)

    return predicted, decfxn
###############################################################################


def markup(predicted, decfxn, data, scrapedhtml):
    """
    Mark up HTML for sentences predicted to be highlighted by the model.

    Args:
    - predicted (array): predicted category for each sentence
    - decfxn (array): confidence score for each sentence
    - data (df): dataframe with feature values for each sentence
    - scrapedhtml (str): full HTML from webpage

    Returns:
    - htmlmarkup (BeautifulSoup): Beautiful Soup object for marked-up HTML
    """

    soup = BeautifulSoup(scrapedhtml, 'lxml')

    predict = list(predicted)
    tmpsoup = str(soup)
    decision = list(decfxn)
    n = 0

    for f in data['sentences']:
        if predict[n] == 1:
            if decision[n] >= 0.1:
                # Mark up HTML to highlight sentence
                newf = '<span style="background-color: #ffff00">'+f+'</span>'
                tmpsoup = tmpsoup.replace(f, newf)

        n += 1

    outsoup = BeautifulSoup(tmpsoup, 'lxml')
    htmlmarkup = outsoup.prettify()

    return htmlmarkup
###############################################################################


@app.route('/')
@app.route('/index')
def index():
    return render_template("index.html")


@app.route('/output')
def output():
    inputtext = request.args.get('inputtext')
    if not inputtext.startswith('http'):
        return render_template('error.html')
    scrapedhtml = webscrape(inputtext)
    cleanedtext = getfulltext(scrapedhtml)
    data = calc_params(cleanedtext)
    predicted, decfxn = predict(data)
    htmlmarkup = markup(predicted, decfxn, data, scrapedhtml)
    return render_template('output.html', html=htmlmarkup)
