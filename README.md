# Employee Sentiment Analysis Project

**Author:** Bhavesh Mantrabuddi 
**Date:** 1/20/2026 
**Language:** Python 3  

## 1. Project Overview
This project delivers an engineered Natural Language Processing (NLP) solution to analyze employee communications at Glynac. By leveraging a rule-based sentiment engine and statistical modeling, the project aims to quantify employee engagement, identify potential retention risks ("flight risks"), and predict future sentiment trends.

## 2. Methodology
The solution is broken down into four distinct phases:
* **Phase 1: Sentiment Engineering:** Implementation of a custom, context-aware sentiment scoring algorithm that accounts for negation (*"not happy"*) and intensity (*"very bad"*).
* **Phase 2: Exploratory Data Analysis (EDA):** Visualization of sentiment distribution and time-series trends using Seaborn and Matplotlib.
* **Phase 3: Employee Analytics:** Aggregation of scores to rank top performers and algorithmically detect flight risks based on negative communication bursts.
* **Phase 4: Predictive Modeling:** Development of a Linear Regression model (`scikit-learn`) to predict monthly sentiment scores based on communication volume and length.

## 3. Key Findings

### Employee Rankings
* **Top Positive Employees (Highest Engagement):**
    1. Lydia Delgado (`lydia.delgado@enron.com`) - Score: 92
    2. Patti Thompson (`patti.thompson@enron.com`) - Score: 91
    3. Johnny Palmer (`johnny.palmer@enron.com`) - Score: 87
    
* **Concern Areas (Lowest Total Sentiment Scores):**
    1. Kayne Coulter (`kayne.coulter@enron.com`) - Score: 51
    2. Rhonda Denton (`rhonda.denton@enron.com`) - Score: 53
    3. Don Baughman (`don.baughman@enron.com`) - Score: 63

### Flight Risks identified
The system flags employees who send **3+ negative emails within a rolling 30-day window**.
* **Flagged Employees:**
    * `john.arnold@enron.com`
    * `eric.bass@enron.com`
*(Note: These employees exhibited specific bursts of negative communication meeting the density threshold, despite their overall standing.)*

### Model Performance
* **Model:** Linear Regression
* **R² Score:** 0.53 (Moderate predictive power)
* **MSE:** 2.37
* **Insight:** The model indicates that **Email Volume** (Coef: 0.2745) is the strongest predictor of total sentiment score. This suggests that in this dataset, frequent communicators tend to accumulate higher positive engagement scores, while message length (Coef: 0.0020) has a negligible impact.

## 4. Repository Structure
The project is modularized for maintainability and reproducibility:

```text
/Project_Root
│
├── main_analysis.ipynb       # The primary report containing execution, plots, and narrative.
├── src/                      # Source code package
│   ├── __init__.py
│   ├── sentiment.py          # Custom NLP logic (handling negation/intensity).
│   └── analytics.py          # Functions for ranking and flight risk detection.
├── visuals/                  # Output folder for generated charts (EDA).
├── test(in).csv              # Raw input dataset.
├── test_augmented.csv        # Processed dataset with 'Sentiment' and 'Score' columns.
└── requirements.txt          # Python dependencies.
