"""
Handles feature engineering and Linear Regression modeling using scikit-learn.
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

def prepare_features(df):
    """
    Engineers features for the model by aggregating data to a monthly level.
    
    Creates:
    - email_count: Number of emails sent per month.
    - avg_msg_len: Average length of messages sent per month.
    - Sentiment_Score: Target variable (sum of scores).
    """
    # Ensure date is datetime and create Month_Year
    df = df.copy()
    if 'date' not in df.columns:
        raise ValueError("Dataframe must contain a 'date' column.")
        
    df['Month_Year'] = df['date'].dt.to_period('M')
    df['msg_len'] = df['body'].astype(str).str.len()

    # Aggregate by Employee and Month
    monthly_data = df.groupby(['from', 'Month_Year']).agg({
        'Sentiment_Score': 'sum',
        'body': 'count',
        'msg_len': 'mean'
    }).reset_index()

    monthly_data.rename(columns={'body': 'email_count', 'msg_len': 'avg_msg_len'}, inplace=True)

    # Define Features (X) and Target (y)
    X = monthly_data[['email_count', 'avg_msg_len']]
    y = monthly_data['Sentiment_Score']

    return monthly_data, X, y

def train_and_evaluate(X, y):
    """
    Trains a Linear Regression model and returns performance metrics.
    """
    # Train-Test Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Model Training
    model = LinearRegression()
    model.fit(X_train, y_train)
    preds = model.predict(X_test)

    # Calculate Metrics
    mse = mean_squared_error(y_test, preds)
    r2 = r2_score(y_test, preds)
    
    metrics = {
        'MSE': mse,
        'R2': r2,
        'Coef_Email_Count': model.coef_[0],
        'Coef_Msg_Len': model.coef_[1]
    }

    return model, metrics