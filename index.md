# Portfolio
---
## Natural Language Processing

### Summarization and translation from YouTube videos using mT5 multilanguage tranformers architecture.

[![View on GitHub](https://img.shields.io/badge/GitHub-View_on_GitHub-blue?logo=GitHub)](https://github.com/chriskhanhtran/CS224n-NLP-Solutions/tree/master/assignments/)

**Summarization:** In this notebook I'll use the HuggingFace's transformers library to fine-tune a pretrained summarization module which summarize texts in Spanish using the fine-tuned multilanguage mT5 encoder/decorder architecture.
<br>
[![Run in Google Colab](https://img.shields.io/badge/Colab-Run_in_Google_Colab-blue?logo=Google&logoColor=FDBA18)](https://colab.research.google.com/drive/1f32gj5IYIyFipoINiC8P3DvKat-WWLUK)


**Neural Machine Translation:** In this notebook I'll use the HuggingFace's transformers library to fine-tune a pretrained NMT module which translates texts from Spanish to English using the multilanguage mT5 encoder/decorder architechture.
<br>
[![Run in Google Colab](https://img.shields.io/badge/Colab-Run_in_Google_Colab-blue?logo=Google&logoColor=FDBA18)](https://colab.research.google.com/drive/1f32gj5IYIyFipoINiC8P3DvKat-WWLUK)

**Flask End-to-End Deep learning APP:** Front-end and Back-end using the pretrained models and deployed with Flask 
([GitHub](https://github.com/chriskhanhtran/CS224n-NLP-Solutions/tree/master/assignments/)).


<center><img src="images/t5.png"/></center>

---

### Simple-Movie-Recommender from IMDB movies dataset

[![View on GitHub](https://img.shields.io/badge/GitHub-View_on_GitHub-blue?logo=GitHub)](https://github.com/andresca94/Simple-Movie-Recommender)

<div style="text-align: justify">This project cover 2 approaches, the first simple recommender does the following: Decide on the metric or score to rate movies on. Calculate the score for every movie. Sort the movies based on the score and output the top results. The second approach build a system that recommends movies that are similar to a particular movie to finally create the recommendation based on the following metadata: the 3 top actors, the director, related genres, and the movie plot keywords. Compute pairwise cosine similarity scores for all movies based on their plot descriptions and recommend movies based on that similarity score threshold. Tokenized, vectorized text data using TF-IDF and cosine similarity to get a content based recommender and credits, genres, and keywords based recommender.</div>
<br>
<center><img src="images/recom1.PNG" width = 50%></center>
<center><img src="images/recomend2.PNG"></center>
<br>

---
### Detect Spam Messages: TF-IDF and Naive Bayes Classifier

[![Open Notebook](https://img.shields.io/badge/Jupyter-Open_Notebook-blue?logo=Jupyter)](projects/detect-spam-nlp.html)
[![View on GitHub](https://img.shields.io/badge/GitHub-View_on_GitHub-blue?logo=GitHub)](https://github.com/chriskhanhtran/detect-spam-messages-nlp/blob/master/detect-spam-nlp.ipynb)

<div style="text-align: justify">In order to predict whether a message is spam, first I vectorized text messages into a format that machine learning algorithms can understand using Bag-of-Word and TF-IDF. Then I trained a machine learning model to learn to discriminate between normal and spam messages. Finally, with the trained model, I classified unlabel messages into normal or spam.</div>
<br>
<center><img src="images/detect-spam-nlp.png"/></center>
<br>

---
## Data Science

### MasterCard stock price time series forecasting using LSTM and GRU

[![View on GitHub](https://img.shields.io/badge/GitHub-View_on_GitHub-blue?logo=GitHub)](https://github.com/chriskhanhtran/CS224n-NLP-Solutions/tree/master/assignments/)

<div style="text-align: justify">RNN remembers past inputs due to an internal memory which is useful for predicting stock prices, generating text, transcriptions, and machine translation. In this notebook I'll use LSTM and GRU that are an advanced type of RNN. RNN simple structure suffers from short memory, where it struggles to retain previous time step information in larger sequential data. These problems can easily be solved by long short term memory (LSTM) and gated recurrent unit (GRU), as they are capable of remembering long periods of information. In this notebook I will show you a preprocessed dataset from May-25-2006 to Oct-11-202 and how to built machine learning models to predict the stock price using both LSTM and GRU.</div>

<center><img src="images/RNN.png"/></center>

<center><img src="images/Merged_document.png"/></center>

<center><img src="images/Merged_document (1).png"/></center>

---

### Credit Risk Prediction Web App

[![Open Web App](https://img.shields.io/badge/Heroku-Open_Web_App-blue?logo=Heroku)](http://credit-risk.herokuapp.com/)
[![Open Notebook](https://img.shields.io/badge/Jupyter-Open_Notebook-blue?logo=Jupyter)](https://github.com/chriskhanhtran/credit-risk-prediction/blob/master/documents/Notebook.ipynb)
[![View on GitHub](https://img.shields.io/badge/GitHub-View_on_GitHub-blue?logo=GitHub)](https://github.com/chriskhanhtran/credit-risk-prediction)

<div style="text-align: justify">After my team preprocessed a dataset of 10K credit applications and built machine learning models to predict credit default risk, I built an interactive user interface with Streamlit and hosted the web app on Heroku server.</div>
<br>
<center><img src="images/credit-risk-webapp.png"/></center>
<br>

---
### Kaggle Competition: Predict Ames House Price using Lasso, Ridge, XGBoost and LightGBM

[![Open Notebook](https://img.shields.io/badge/Jupyter-Open_Notebook-blue?logo=Jupyter)](projects/ames-house-price.html)
[![View on GitHub](https://img.shields.io/badge/GitHub-View_on_GitHub-blue?logo=GitHub)](https://github.com/chriskhanhtran/kaggle-house-price/blob/master/ames-house-price.ipynb)

<div style="text-align: justify">I performed comprehensive EDA to understand important variables, handled missing values, outliers, performed feature engineering, and ensembled machine learning models to predict house prices. My best model had Mean Absolute Error (MAE) of 12293.919, ranking <b>95/15502</b>, approximately <b>top 0.6%</b> in the Kaggle leaderboard.</div>
<br>
<center><img src="images/ames-house-price.jpg"/></center>
<br>

---
### Predict Breast Cancer with RF, PCA and SVM using Python

[![Open Notebook](https://img.shields.io/badge/Jupyter-Open_Notebook-blue?logo=Jupyter)](projects/breast-cancer.html)
[![View on GitHub](https://img.shields.io/badge/GitHub-View_on_GitHub-blue?logo=GitHub)](https://github.com/chriskhanhtran/predict-breast-cancer-with-rf-pca-svm/blob/master/breast-cancer.ipynb)

<div style="text-align: justify">In this project I am going to perform comprehensive EDA on the breast cancer dataset, then transform the data using Principal Components Analysis (PCA) and use Support Vector Machine (SVM) model to predict whether a patient has breast cancer.</div>
<br>
<center><img src="images/breast-cancer.png"/></center>
<br>

---
### Business Analytics Conference 2018: How is NYC's Government Using Money?

[![Open Research Poster](https://img.shields.io/badge/PDF-Open_Research_Poster-blue?logo=adobe-acrobat-reader&logoColor=white)](pdf/bac2018.pdf)

<div style="text-align: justify">In three-month research and a two-day hackathon, I led a team of four students to discover insights from 6 million records of NYC and Boston government spending data sets and won runner-up prize for the best research poster out of 18 participating colleges.</div>
<br>
<center><img src="images/bac2018.JPG"/></center>
<br>

---
## More about me...

<div style="text-align: justify">Besides Data Science, I also have a great passion for soccer and music. I love to see great games and live concerts as well I try to play soccer very often and play guitar and piano.</div>
<br>
---
<center>© 2023 Andres Carvajal. Powered by Jekyll and the Minimal Theme.</center>
