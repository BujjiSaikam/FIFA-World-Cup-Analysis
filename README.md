# FIFA World Cup Analysis

## Introduction
The FIFA World Cup is the most prestigious football tournament in the world, held every four years and featuring national teams from across the globe. This project aims to analyze historical FIFA World Cup data to uncover key metrics and factors influencing World Cup outcomes. By examining various dimensions such as match attendance, goals scored, team performance, and player statistics, we aim to derive insights that shed light on the tournament's trends and patterns.

### Objectives:
- Explore and visualize key metrics related to FIFA World Cup tournaments.
- Compare the performance of winners, runners-up, and third-place teams.
- Analyze attendance trends, goal statistics, and match distributions.
- Evaluate player performance metrics and identify top players.
- Build a machine learning model to predict match outcomes.

### Data Source:
- **WorldCups.csv**: Information about all FIFA World Cups.
- **WorldCupMatches.csv**: Results from matches contested in the World Cups.
- **WorldCupsPlayers.csv**: Player statistics from the World Cups.

## Table of Contents
1. [Introduction](#introduction)
2. [Objectives](#objectives)
3. [Data Source](#data-source)
4. [Tools and Libraries](#tools-and-libraries)
5. [Data Preparation](#data-preparation)
6. [Exploratory Data Analysis (EDA)](#exploratory-data-analysis-eda)
7. [Machine Learning](#machine-learning)
8. [Key Insights](#key-insights)
9. [Conclusion](#conclusion)
10. [Future Work](#future-work)

## Tools and Libraries
- **Python**: Primary programming language for data analysis and visualization.
- **Pandas**: For data manipulation and cleaning.
- **NumPy**: For numerical operations.
- **Matplotlib & Seaborn**: For data visualization.
- **Scikit-learn**: For machine learning model building and evaluation.
- **SQL**: For data querying and management.

### Installation Guide
1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/FIFA-World-Cup-Analysis.git
   cd FIFA-World-Cup-Analysis
   ```
2. Create and activate a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows use `venv\Scripts\activate`
   ```
3. Install the required libraries:
   ```bash
   pip install -r requirements.txt
   ```

## Data Preparation
1. **Loading Data**: Import the datasets and inspect their structure.
2. **Data Cleaning**: Handle missing values and remove duplicates to ensure data quality.
   
   ```python
   import pandas as pd
   import numpy as np

   # Load the datasets
   world_cups = pd.read_csv('WorldCups.csv')
   matches = pd.read_csv('WorldCupMatches.csv')
   players = pd.read_csv('WorldCupPlayers.csv')

   # Data cleaning
   world_cups.dropna(inplace=True)
   matches.dropna(inplace=True)
   players.dropna(inplace=True)
   
   world_cups.drop_duplicates(inplace=True)
   matches.drop_duplicates(inplace=True)
   players.drop_duplicates(inplace=True)
   ```

## Exploratory Data Analysis (EDA)
- Analyze the performance of winners, runners-up, and third-place teams.
- Visualize the distribution of matches across host cities.
- Examine attendance trends over the years.
- Analyze goal-scoring trends across tournaments.
- Evaluate the number of teams participating in each tournament.
- Assess average match attendance year by year.
- Identify stadiums with the highest average attendance.
- Compare the distribution of goals scored by home and away teams.
- Identify the top goal scorers in World Cup history.
- Evaluate player performance metrics such as goals scored and matches played.

## Machine Learning
1. **Predictive Modeling**: Build a machine learning model to predict match outcomes based on features like goals scored, attendance, and more.
2. **Model Training and Evaluation**: Train a Random Forest Classifier and Hist Gradient Boosting Classifier and evaluate their accuracy.

   ```python
   from sklearn.impute import SimpleImputer
   from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
   from sklearn.metrics import accuracy_score
   from sklearn.model_selection import train_test_split

   # Prepare features and labels
   features = match[['Home Team Goals', 'Away Team Goals', 'Attendance_y']]
   labels = match['Home Team Name'] == match['Winner']

   # Impute missing values
   imputer = SimpleImputer(strategy='mean')
   X_imputed = imputer.fit_transform(features)

   # Split data into training and testing sets
   X_train, X_test, y_train, y_test = train_test_split(X_imputed, labels, test_size=0.2, random_state=42)

   # Train and evaluate models
   rf_model = RandomForestClassifier()
   rf_model.fit(X_train, y_train)
   accuracy_rf = accuracy_score(y_test, rf_model.predict(X_test))

   hist_model = HistGradientBoostingClassifier()
   hist_model.fit(X_train, y_train)
   accuracy_hist = accuracy_score(y_test, hist_model.predict(X_test))

   print(f"Accuracy of RandomForestClassifier: {accuracy_rf:.2f}")
   print(f"Accuracy of HistGradientBoostingClassifier: {accuracy_hist:.2f}")
   ```

## Key Insights
- Visualization of team performances highlights dominant teams in World Cup history.
- Attendance analysis reveals trends and fluctuations over the years.
- Goal analysis helps understand scoring patterns and influential players.
- Machine learning model provides a predictive framework for match outcomes.

## Conclusion
This project provides a comprehensive analysis of FIFA World Cup data, uncovering significant trends and patterns. By leveraging data visualization and machine learning techniques, we gain deeper insights into the tournament's dynamics. The analysis not only highlights historical trends but also offers predictive insights for future tournaments.

## Future Work
- Incorporate more advanced machine learning models.
- Add more features for predictive modeling.
- Extend the analysis to include more recent tournaments.

Thank you for your time!
