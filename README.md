Women's Clothing E-Commerce Project

Title: Women's Clothing E-Commerce Reviews Analysis
Authors: Sanaullah Shafaq, Nurgazt Dias Aslanuly
Institution: CS
Course: Data Mining
Date: 05/12/2025

Project Overview

This project provides a comprehensive data analysis and modeling workflow for the Women's Clothing E-Commerce Reviews dataset. The objective is to explore customer reviews, analyze patterns in ratings and recommendations, perform NLP on textual reviews, build predictive models, and generate actionable insights.

Key components include:

Data preprocessing and cleaning

Exploratory Data Analysis (EDA)

Statistical analysis and hypothesis testing

Machine learning: supervised and unsupervised models

Time series forecasting

Neural network modeling

Natural Language Processing (NLP)

Ethics and data security considerations

Dataset

Source: Kaggle / CSV file Womens Clothing E-Commerce Reviews.csv

Number of records: X

Number of variables: Y

Data types:

Numerical: Rating, Age, Positive Feedback Count

Categorical: Department Name, Division Name, Class Name

Text: Review Text

Project Structure

The project follows a full data lifecycle: data collection → preprocessing → analysis → modeling → evaluation.

Key Steps:

Data Preprocessing

Fill missing values (Review Text, Recommended IND)

Remove duplicates

Normalize numeric features

Encode categorical features

Split data into training/testing sets

Exploratory Data Analysis (EDA)

Distribution analysis: Rating, Age, Recommendation

Visualizations: histograms, pie charts, barplots, boxplots

Correlation analysis of numeric features

Statistical Analysis

ANOVA tests across top departments

Insights on rating differences by categories

Machine Learning Models

Supervised Learning: TF-IDF Logistic Regression (text), Random Forest (numeric)

Metrics: Accuracy, Precision, Recall, F1-score

Unsupervised Learning: KMeans clustering and PCA on numeric features

Time Series Analysis

Synthetic monthly review counts

Seasonal decomposition and ARIMA forecasting

Neural Network

Simple dense network trained on numeric features

Comparison with Random Forest and Logistic Regression models

Natural Language Processing (NLP)

Text cleaning and tokenization

TF-IDF vectorization

Word cloud visualization

Sentiment analysis

Ethics and Data Security

Bias detection (age, department)

Fairness assessment for recommendations

User privacy considerations

Outputs and Figures

The project generates multiple visualizations and tables for analysis:

Figures: Ratings distribution, Age distribution, Correlation heatmaps, Boxplots, Confusion matrices, PCA plots, Word clouds, Neural Network diagrams, Model performance comparison

Tables: Ethical risk analysis, Model results

Results and Insights

TF-IDF Logistic Regression performs well on textual review classification.

Random Forest and Neural Network models provide comparable results on numeric features.

EDA revealed trends in customer recommendations by department and rating distributions.

Sentiment analysis highlights positive and negative feedback patterns.

Conclusion and Future Work

Learned valuable insights from customer reviews and ratings.

Predictive models show potential for recommendation systems.

Future improvements:

Apply deep learning for full-text sentiment analysis

Hyperparameter tuning for better ML and NN performance

Expand dataset for more robust modeling

References and Libraries

Dataset: Kaggle "Women's Clothing E-Commerce Reviews"

Libraries: pandas, numpy, matplotlib, seaborn, scikit-learn, nltk, tensorflow, wordcloud, statsmodels

Documentation: scikit-learn, TensorFlow, statsmodels
