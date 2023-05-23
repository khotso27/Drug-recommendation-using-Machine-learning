## Data Cleaning

The script performs several data cleaning steps, including converting column names to lowercase, sorting the dataframe, handling missing values in the 'condition' column by replacing NaN values with empty strings, and removing special characters from the 'review' column.

## Exploratory Data Analysis

The script includes various data visualization techniques to explore the dataset:

1. Boxplot: Visualizes the distribution of the numerical columns 'rating' and 'usefulcount' using a boxplot.

2. Barplots: Shows the top 20 drugs with ratings of 10/10 and 1/10 using barplots. Also, displays a countplot of the 'rating' column and a barplot of the mean rating per year.

3. Word Cloud: Generates word clouds of reviews with ratings equal to 10 and 1, showing the most frequent words in the reviews.

4. Pie Charts: Represents the share of each rating and the sentiments of the patients using pie charts.

5. Barplot: Shows the top 10 conditions that people are suffering from.

6. Histogram: Displays the distribution of the 'usefulcount' column using a histogram.

7. Heatmap: Shows the correlation matrix of the numerical columns using a heatmap.

8. Ngrams: Displays the top 20 unigrams, bigrams, and trigrams according to ratings.

9. Word Count Plot: Plots the word count of the reviews after removing stop words.

## Sentiment Analysis

The script performs sentiment analysis on the ratings and assigns a sentiment value of 1 for ratings greater than 5 (positive sentiment) and 0 for ratings less than or equal to 5 (negative sentiment). It also creates a new column 'sentiment_rate' to store the sentiment values.

This code provides a comprehensive analysis of the dataset, including data cleaning, visualization, and sentiment analysis. It can be used as a starting point for further analysis or as a reference for similar projects

## Code Description

This code performs feature engineering and builds a machine learning model to predict the sentiment rate of drug reviews. Here's a breakdown of the main sections of the code:

### 1. Feature Engineering

In this section, several features are engineered from the data. The following features are created:

### 2. Label Encoding

In this section, the categorical features "drugname" and "condition" are label encoded using sklearn's LabelEncoder. This process assigns a numerical value to each unique category, enabling the machine learning model to work with categorical data.

### 3. Modelling

In this section, the machine learning model is defined and trained using the XGBoost algorithm. The dataset is split into training and testing sets, and the XGBoost classifier is initialized with specified hyperparameters. The model is then trained on the training set and used to make predictions on the test set.

### 4. Evaluation

After making predictions, the code evaluates the performance of the model by calculating the accuracy score and generating a confusion matrix. The accuracy score represents the percentage of correct predictions made by the model. The confusion matrix provides a detailed breakdown of the true positive, true negative, false positive, and false negative predictions.

### 5. Feature Importance Plot

In this section, a feature importance plot is generated using the XGBoost classifier. This plot shows the relative importance of each feature in predicting the sentiment rate. It helps in understanding which features have the most significant impact on the model's predictions.

## Usage

To use this code, follow these steps:

1. Ensure that you have all the necessary libraries installed, including xgboost, lightgbm, catboost, and sklearn.
2. Prepare your dataset with the required columns ('condition', 'usefulcount', 'sentiment', 'day', 'month', 'year', 'sentiment_clean_ss', 'count_word', 'count_unique_word', 'count_letters', 'count_punctuations', 'count_words_upper', 'count_words_title', 'count_stopwords', 'mean_word_len') and the target column ('sentiment_rate').
3. Replace the variable names in the code with the corresponding columns in your dataset.
4. Run the code and observe the outputs, including the accuracy score, confusion matrix, and feature importance plot.

## Dependencies

This code requires the following dependencies:
#Libraries to be used for our model

- pandas as pd
- numpy as np
- seaborn as sns
- matplotlib.pyplot as plt
- wordcloud import WordCloud
- textblob import TextBlob
- nltk.corpus import stopwords
- collections import Counter
- warnings; warnings.simplefilter('ignore')
- nltk
- string
- nltk import ngrams
- nltk.tokenize import word_tokenize 
- nltk.stem import SnowballStemmer
- xgboost
- lightgbm
- catboost
- sklearn

Make sure these libraries are installed before running the code.

Note: This code assumes that the dataset has already been preprocessed and cleaned. If not, additional preprocessing steps may be required before using this code.

## Acknowledgments

This code was developed based on a specific dataset and task. Credit goes to the original authors of the code and the dataset used. If applicable, please acknowledge and cite the original sources when using this code for your own projects.

## Disclaimer

This code is provided as-is and without any warranty. The authors shall not be