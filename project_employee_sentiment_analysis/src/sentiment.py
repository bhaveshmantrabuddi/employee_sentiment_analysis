"""
NLP Logic
Context-aware sentiment analysis. Includes logic for handling negation (e.g., "not happy") and intensity modifiers (e.g., "very bad").
"""

import re

def get_sentiment(message):
    """
    Analyzes a text string and returns 'Positive', 'Negative', or 'Neutral'.
    Uses a bag-of-words approach modified by negation and intensity lookaheads.
    """
    
    message = str(message).lower()

    # --- CUSTOM LEXICONS ---
    positive_words = ['great', 'good', 'excellent', 'happy', 'positive', 'successful', 'appreciate', 'thank', 'thanks',
                      'wonderful', 'pleasure', 'agree', 'awesome', 'best', 'effective', 'improve', 'support', 'resolve',
                      'progress', 'opportunity', 'strong', 'confident', 'efficient', 'benefit', 'forward']
    negative_words = ['bad', 'poor', 'negative', 'unsuccessful', 'concern', 'issue', 'problem', 'difficult', 'unfortunately',
                      'unable', 'deny', 'cancelled', 'delay', 'crisis', 'error', 'failed', 'failure', 'stress', 'terrible',
                      'trouble', 'wrong', 'reject', 'decline', 'risk', 'disappoint', 'complain', 'frustrate', 'worry']
    negation_words = ['not', 'no', 'never', 'none', 'nor', 'hardly', 'barely', 'scarcely', "don't", "doesn't", "didn't",
                      "isn't", "aren't", "wasn't", "weren't", "haven't", "hasn't", "hadn't", "won't", "wouldn't",
                      "can't", "couldn't", "shouldn't", "mightn't", "mustn't"]
    intensity_words = {'very': 2.0, 'extremely': 2.5, 'really': 1.8, 'quite': 0.8, 'slightly': 0.5}

    words = re.findall(r'\b\w+\b', message)
    score = 0
    i = 0
    while i < len(words):
        word = words[i]
        current_multiplier = 1.0
        negate_next = False
        
        # Check negation
        if word in negation_words:
            negate_next = True
            i += 1
            if i >= len(words): break
            word = words[i]
            
        # Check intensity
        if word in intensity_words:
            current_multiplier = intensity_words[word]
            i += 1
            if i >= len(words): break
            word = words[i]
            
        # Value Assignment
        val = 0
        if word in positive_words: val = 1
        elif word in negative_words: val = -1
        
        if negate_next: val *= -1
        val *= current_multiplier
        score += val
        i += 1
        
    if score > 0: return 'Positive'
    elif score < 0: return 'Negative'
    else: return 'Neutral'