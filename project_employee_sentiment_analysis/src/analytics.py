'''
Consider ranking and flight risk logic/helper functions for aggregating employee scores and detecting retention risks.
'''


import pandas as pd

def get_employee_rankings(df):
    # Returns top 3 positive and top 3 negative employees based on total score.
    employee_scores = df.groupby('from')['Sentiment_Score'].sum().sort_values(ascending=False)
    return employee_scores.head(3), employee_scores.tail(3)

def detect_flight_risks(df):
    # Identifies employees who sent 3+ negative emails within any 30-day rolling window.
    # Ensure data is sorted for window calculation
    df = df.sort_values(['from', 'date'])
    negative_emails = df[df['Sentiment'] == 'Negative']
    
    flight_risks = []
    
    for employee, group in negative_emails.groupby('from'):
        if len(group) >= 3:
            # Sliding window check for every sequence of 3 emails
            for i in range(len(group) - 2):
                start = group.iloc[i]['date']
                end = group.iloc[i+2]['date']
                # If the span between the 1st and 4th negative email is <= 30 days
                if (end - start).days <= 30:
                    flight_risks.append(employee)
                    break
                    
    return list(set(flight_risks))