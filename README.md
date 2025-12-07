## Women's Clothing E-Commerce Reviews Analysis

Authors: Sanaullah Shafaq, Nurgazt Dias Aslanuly
Institution: CS
Course: Data Mining
Date: 05/12/2025

🚀 Project Overview

This project provides a full data mining workflow for the Women's Clothing E-Commerce Reviews dataset. The analysis includes:

Data cleaning and preprocessing

Exploratory Data Analysis (EDA)

Statistical analysis and hypothesis testing

Machine Learning (supervised and unsupervised)

Time series forecasting

Neural Network modeling

Natural Language Processing (NLP)

Ethical considerations and data security

The goal is to extract insights, predict recommendations, and explore customer sentiment from reviews.

📂 Dataset

Source: Kaggle / CSV file: Womens Clothing E-Commerce Reviews.csv

Records: X

Variables: Y

Data types:

Numerical: Rating, Age, Positive Feedback Count

Categorical: Department Name, Division Name, Class Name

Text: Review Text

🛠 Project Structure

The project follows the full data lifecycle: collection → preprocessing → analysis → modeling → evaluation.

1. Data Preprocessing

Fill missing values (Review Text, Recommended IND)

Remove duplicates

Normalize numeric features

Encode categorical features

Split data into train/test sets

2. Exploratory Data Analysis (EDA)

Statistical summary and feature distributions

Visualizations: histograms, pie charts, barplots, boxplots

Correlation heatmaps for numeric variables

3. Statistical Analysis

ANOVA tests across top departments

Key insights on rating differences by categories

4. Machine Learning Models

Supervised Learning:

TF-IDF + Logistic Regression (text features)

Random Forest (numeric features)

Metrics: Accuracy, Precision, Recall, F1-score

Unsupervised Learning:

KMeans clustering on numeric features

PCA for dimensionality reduction

5. Time Series Analysis

Synthetic monthly review counts

Seasonal decomposition and ARIMA forecasting

6. Neural Network

Dense network with input, hidden, and output layers

Trained on numeric features for 10 epochs

Comparison with traditional ML models

7. Natural Language Processing (NLP)

Text cleaning, tokenization, and TF-IDF vectorization

Word cloud visualization

Sentiment analysis of reviews

8. Ethics and Data Security

Bias detection (age, department categories)

Fairness analysis for recommendations

Privacy considerations for user reviews

📊 Figures & Visualizations

Figure 1: Data lifecycle

Figure 2: Dataset structure & variable types

Figure 3: Missing values before & after cleaning

Figure 4: Normalized numerical features

Figure 5: Correlation heatmap

Figure 6: Boxplots of feature distributions

Figure 7: Linear regression & confidence intervals

Figure 8: Confusion matrix for classification results

Figure 9: PCA 2D visualization of clusters

Figure 10: Time series forecast (Predicted vs Actual)

Figure 11: Neural Network architecture diagram

Figure 12: Training loss & accuracy curves

Figure 13: Word cloud of reviews

Figure 14: Sentiment distribution chart

Figure 15: Model performance comparison

Tip: Save figures in /figures/ and embed them using markdown:
![Figure 1](figures/fig1.png)

📈 Results & Insights

TF-IDF + Logistic Regression performs well for text classification.

Random Forest and Neural Network provide comparable results for numeric features.

EDA shows patterns in ratings, recommendations, and departmental differences.

NLP analysis highlights sentiment trends across customer reviews.

📝 Conclusion & Future Work

Learned patterns in customer behavior, sentiment, and predictive modeling.

Future improvements:

Apply deep learning on text for sentiment prediction

Hyperparameter tuning for ML & NN models

Use larger datasets for robust modeling

📚 References

Dataset: Kaggle "Women's Clothing E-Commerce Reviews"

Libraries: pandas, numpy, matplotlib, seaborn, scikit-learn, nltk, tensorflow, wordcloud, statsmodels

Documentation: scikit-learn, TensorFlow, statsmodels
