# Housing Price Prediction Using Machine Learning

## Project Overview

Housing Price Prediction is a machine learning regression project that predicts the price of a house based on features such as location, area, number of bedrooms, number of bathrooms, and other property-related details.

The main objective is to train a machine learning model using historical housing data and predict prices for new houses.

## Objective

- Understand housing data and its features.
- Perform data cleaning and preprocessing.
- Explore relationships between house features and prices.
- Train regression models to predict house prices.
- Evaluate model performance using appropriate metrics.

## Types of Machine Learning

Housing Price Prediction is a **supervised machine learning regression problem** because the model learns from labeled data containing house features and their actual prices.

## Workflow

1. Import the required libraries.
2. Load the housing dataset.
3. Explore the dataset using Pandas.
4. Handle missing values and duplicate records.
5. Perform exploratory data analysis.
6. Separate independent and dependent variables.
7. Encode categorical variables if required.
8. Split the dataset into training and testing sets.
9. Train regression models.
10. Evaluate the model and predict housing prices.

## Algorithms Used

### 1. Linear Regression

Linear Regression predicts house prices by identifying a linear relationship between input features and the target variable.

**Advantages**
- Simple and easy to understand.
- Fast to train.
- Provides an interpretable baseline model.

**Disadvantages**
- Assumes a linear relationship between features and price.
- May not perform well when relationships are complex.
- Can be affected by outliers.

### 2. Decision Tree Regressor

A Decision Tree Regressor divides the dataset into smaller groups based on feature conditions and predicts a price for each resulting region.

**Advantages**
- Can capture nonlinear relationships.
- Easy to understand and visualize.
- Requires little feature scaling.

**Disadvantages**
- Can overfit the training data.
- Small changes in data may produce a different tree.
- May generalize poorly without proper tuning.

### 3. Random Forest Regressor

Random Forest combines multiple decision trees
