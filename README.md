# Amazon-Alexa-Review-Sentiment_Analysis
## Sentiment analysis on amazon product's review  - Amazon Echo White

- Applying the Amazon Alexa dataset and building classification models to predict if the sentiment of a given input sentence is positive or negative. 

## Dataset: 
- Source link: https://www.kaggle.com/datasets/sid321axn/amazon-alexa-reviews/data 

- This dataset consists of a nearly 3000 Amazon customer reviews (input text), star ratings, date of review, variant and feedback of various amazon Alexa products like Alexa Echo, Echo dots, Alexa Firesticks etc. for learning how to train Machine for sentiment analysis.

## What you can do with this Data ?
You can use this data to analyze Amazon’s Alexa product ; discover insights into consumer reviews and assist with machine learning models.You can also train your machine models for sentiment analysis and analyze customer reviews how many positive reviews ? and how many negative reviews ?


### Installing the kaggle library to fetch the data from kaggle: 
    %pip install kaggle 

#### Configuring json file
    !mkdir -p ~/.kaggle 
    !cp kaggle.json ~/.kaggle / 
    !chmod 600 ~/.kaggle/kaggle.json


### API to fetch the dataset from kaggle:
    ! kaggle datasets download -d sid321axn/amazon-alexa-reviews


### Extracting the dataset: 
    from zipfile import ZipFile
    dataset_path = "./amazon-alexa-reviews.zip"
    with ZipFile(dataset_path, "r") as zip:
        zip.extractall(".")
        print("Dataset extracted successfully")



### Installing required dependencies: 
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns
    import nltk
    import pickle 
    import re
    %pip install wordcloud
    %pip install xgboost
    
    from nltk.stem.porter import PorterStemmer
    nltk.download('stopwords')
    from nltk.corpus import stopwords
    STOPWORDS = set(stopwords.words('english'))
    
    
    from sklearn.model_selection import train_test_split
    from sklearn.feature_extraction.text import CountVectorizer 
    from sklearn.preprocessing import MinMaxScaler

    from sklearn.model_selection import cross_val_score
    from sklearn.ensemble import RandomForestClassifier
    
    from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
    from sklearn.model_selection import GridSearchCV
    from sklearn.model_selection import StratifiedKFold
    from sklearn.metrics import accuracy_score
    
    from wordcloud import WordCloud
    from sklearn.tree import DecisionTreeClassifier
    from xgboost import XGBClassifier

## EDA: 
### Models - Select one of the best
    - Random Forest
    - K-fold cross validation + Applying grid search to get the optimal parameters on random forest
    - XGBoost (best one)
    - Decision Tree Classifier 

![Screenshot 2026-02-28 at 11 39 07 PM](https://github.com/user-attachments/assets/ae5ac6db-8c74-4d17-8d3c-503cc81de3c2)

