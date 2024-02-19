# Portfolio
---
## Natural Language Processing

### Summarization and translation from YouTube videos using mT5 multilanguage tranformers architecture.
<center><img src="images/Multilanguage mT5 Summarization and translation APP.gif"/></center>
[![View on GitHub](https://img.shields.io/badge/GitHub-View_on_GitHub-blue?logo=GitHub)](https://github.com/andresca94/Summarization-and-translation-from-YouTube-videos-using-mT5-multilanguage-tranformers-architecture.)
<center><img src="images/t5.png"/></center>
**Summarization:**
<div style="text-align: justify">In this notebook I'll use the HuggingFace's transformers library to fine-tune a pretrained summarization module which summarize texts in Spanish using the fine-tuned multilanguage mT5 encoder/decorder architecture.</div>
<br>
[![Run in Google Colab](https://img.shields.io/badge/Colab-Run_in_Google_Colab-blue?logo=Google&logoColor=FDBA18)](https://colab.research.google.com/drive/11jG-rhploTYvfrE8P5cUd6LSOTKZcyNX#scrollTo=ZGXDBmcwcxQL)


**Neural Machine Translation:** 
<div style="text-align: justify">In this notebook I’ll use the HuggingFace’s transformers library to fine-tune a pretrained NMT module which translates texts from Spanish to English using the multilanguage mT5 encoder/decorder architechture.</div>
<br>
[![Run in Google Colab](https://img.shields.io/badge/Colab-Run_in_Google_Colab-blue?logo=Google&logoColor=FDBA18)](https://colab.research.google.com/drive/1RbtfwyhdZx8aTWruQMgmuVNXfN0UsQq6?authuser=1)

**Flask End-to-End Deep learning APP:** Front-end and Back-end using the pretrained models and deployed with Flask
<br>
[![Run in Google Colab](https://img.shields.io/badge/Colab-Run_in_Google_Colab-blue?logo=Google&logoColor=FDBA18)](https://colab.research.google.com/drive/1nTsgWxsjUmw2Xiq1PLEfHPY8ZQ0DzPw-?authuser=1#scrollTo=QramYSC3lpa7)


<center><img src="images/ngrok.PNG"/></center>

---

### Simple-Movie-Recommender and Content Based-Recommender from IMDB movies dataset

[![View on GitHub](https://img.shields.io/badge/GitHub-View_on_GitHub-blue?logo=GitHub)](https://github.com/andresca94/Simple-Movie-Recommender)

<div style="text-align: justify">This project cover 2 approaches, the first simple recommender does the following: Decide on the metric or score to rate movies on. Calculate the score for every movie. Sort the movies based on the score and output the top results. The second approach build a system that recommends movies that are similar to a particular movie to finally create the recommendation based on the following metadata: the 3 top actors, the director, related genres, and the movie plot keywords. Compute pairwise cosine similarity scores for all movies based on their plot descriptions and recommend movies based on that similarity score threshold. Tokenized, vectorized text data using TF-IDF and cosine similarity to get a content based recommender and credits, genres, and keywords based recommender.</div>
<br>
<center><img src="images/recom1.PNG" width = 400px height = 400px></center>
<center><img src="images/TWORECO.png"></center>
<br>

---
## Full Stack Development / Machine Learning
### Trabook - Your Travel Companion - Vue.js && FastAPI

[![View on GitHub](https://img.shields.io/badge/GitHub-View_on_GitHub-blue?logo=GitHub)](https://github.com/andresca94/Peaku)

<div style="text-align: justify">
Trabook is a travel platform designed to kickstart your journey with a splash of fun. The app features destination exploration, professional travel advice, and a comprehensive tour planning toolkit. It's built using Vue.js to ensure a responsive and user-friendly interface, coupled with a FastAPI backend for high performance and scalability. PostgreSQL is used for efficient data storage, and Docker containers help with seamless deployment. Dive into the world of Trabook and let your travel dreams take flight.
</div>
<center><img src="images/Trabook.gif"></center>
<br>

---
## Data Science

### MasterCard stock price time series forecasting using LSTM and GRU

[![View on GitHub](https://img.shields.io/badge/GitHub-View_on_GitHub-blue?logo=GitHub)](https://github.com/chriskhanhtran/CS224n-NLP-Solutions/tree/master/assignments/)

<div style="text-align: justify">RNN remembers past inputs due to an internal memory which is useful for predicting stock prices, generating text, transcriptions, and machine translation. In this notebook I'll use LSTM and GRU that are an advanced type of RNN. RNN simple structure suffers from short memory, where it struggles to retain previous time step information in larger sequential data. These problems can easily be solved by long short term memory (LSTM) and gated recurrent unit (GRU), as they are capable of remembering long periods of information. In this notebook I will show you a preprocessed dataset from May-25-2006 to Oct-11-2022 and how to built machine learning models to predict the stock price using both LSTM and GRU.The model consists of either a single hidden layer of LSTM or GRU and an output layer. LSTM units to 125, tanh as activation, the model will train on 50 epochs with 32 batch sizes. GRU model got 5.95 rmse on the test dataset, which is an improvement from the LSTM model with rmse of 6.47.</div>

<center><img src="images/RNN.png"/></center>

<center><img src="images/Merged_document.png"/></center>

<center><img src="images/Merged_document (1).png"/></center>

---

### Multiclass prediction and clustering for music genre
<div style="text-align: justify">Data cleaning, handling entropy, feature engineering and data visualization. Encoding categorical data and preprocessing dataset. Trained XGboots, Random Forest and Logistic Regression, checking accuracy and confusion matrix to select the best model. Feature importance, ROC curve and SHAP values for explainability. Finding the appropriate K-values with elbow method, balancing the dataset after feature selection and evaluating K-means visually with PCA.</div>

**Classification:** 
<br>
[![View on GitHub](https://img.shields.io/badge/GitHub-View_on_GitHub-blue?logo=GitHub)](https://github.com/andresca94/MulticlassPrediction-Music-Genre-Random-Forest-Logistic-Regression-XGBoots)

**Clustering:** 
<br>
[![View on GitHub](https://img.shields.io/badge/GitHub-View_on_GitHub-blue?logo=GitHub)](https://github.com/andresca94/K-means-clustering-for-music-genre-prediction/blob/main/K-means-clustering%20for%20music%20genre%20prediction.ipynb)
<br>
<center><img src="images/Merged_document (3).png"/></center>
<center><img src="images/Merged_document (3).png"/></center>
<br>

---
### Motor-Colission-App-Streamlit

[![View on GitHub](https://img.shields.io/badge/GitHub-View_on_GitHub-blue?logo=GitHub)](https://github.com/andresca94/Motor-Colission-App-Streamlit)

<div style="text-align: justify">I loaded and cleaned the Motor Collision in New York City dataset. Creation of a 3D map and data visualization to respond the questions “Where are the most people injured in NYC?” and “How many collisions occur during a given time of day”. Breakdown by minute and affected type Pedrestrians, Cyclist, Motorist. Front end using Streamlit.</div>
<br>
<center><img src="images/ezgif.com-gif-maker.gif"/></center>

<br>

---
## STEM MSc Civil Engineering Thesis 
[http://hdl.handle.net/1992/52990](https://repositorio.uniandes.edu.co/bitstream/handle/1992/52990/25247.pdf?sequence=1)
<div style="text-align: justify">A program was developed in MATLAB to show the relationship between the probability of occurrence of landslide and the size of the landslide from slope statistics obtained from digital elevation models (DEM) with spatial resolution of (30x30m) of two zones with different geological and geomorphology origins. The program must consider all the possible neighbors or clusters of an elevation matrix as input using image processing tools, and as output the statistics of the related clusters, to conclude if the topography is organized in a fractal way or invariant with the scale and being a characteristic of the size of landslides.</div>
<br>
<center><img src="images/GIF3.gif"/></center>
<center><img src="images/GIF1.gif"/></center>
<br>
---
## More about me...

<div style="text-align: justify">Besides Data Science, I also have a great passion for soccer and music. I love to watch great games and live concerts as well I try to play soccer very often and play guitar and piano.</div>
<br>
---
<br>
<center>© 2023 Andres Carvajal. Powered by Jekyll and the Minimal Theme.</center>
