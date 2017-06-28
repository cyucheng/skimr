# skimr

[skimr](skimr.herokuapp.com) is a tool for automated highlighting of online articles using machine
learning and natural language processing.

## Background

skimr was developed as my project for the Insight Data Science fellowship in Summer 2017.
The rationale for skimr is that reading articles online can be prohibitively time-consuming, 
and it would be useful to have an app that automatically highlights the key parts of an article.
Such an automatic highlighter could be deployed as its own product or used to enhance the
reading experience of article hosting websites.

## Use skimr

skimr is hosted online at [skimr.herokuapp.com](skimr.herokuapp.com). You can paste in the URL of
an article from [Medium.com](https://medium.com), and it will display the article with key sentences
highlighted, based on a machine learning model as described below.

## Development

To choose which sentences in an article to highlight, I decided to train a machine learning model
that could classify each sentence in an article as highlighted or not. This model would be trained
on crowd-sourced highlights from [Medium.com](https://medium.com/), a blog hosting website, which
allows users to highlight sentences in an article that they find interesting or impactful and
displays a top highlight based on popularity.

I generated a dataset of highlighted and non-highlighted sentences by scraping ~3200 popular articles
from Medium.com using Selenium Webdriver in Python, then parsing the html using BeautifulSoup and
natural language processing.

With this dataset, I calculated features, such as a topic similarity score between each sentence
and the article it belongs to (using LDA from [gensim](https://radimrehurek.com/gensim/models/ldamodel.html)),
[readability metrics](https://github.com/mmautner/readability), and sentence position within the article,
and used them to train a logistic regression model. This model was able to predict whether sentences
in the training and test set were highlighted with sensitivities of 63% and 61%, respectively.

The skimr web app uses a frontend written in [Flask](http://flask.pocoo.org/) to accept a URL
input from the user, then scrapes the html, parses the sentences, calculates features, applies the 
model to predict whether it should be highlighted or not, marks up the appropriate sentences in
the html, and finally displays the highlights *in situ* on the article webpage.

For a slide deck describing the development of skimr, see this
[Google Presentation](https://docs.google.com/presentation/d/1UtmQgIopb9BSEu3QvPIi_2M5eXPZ3gavTlrKoZ7Wcow/embed?start=false&loop=false&delayms=3000).