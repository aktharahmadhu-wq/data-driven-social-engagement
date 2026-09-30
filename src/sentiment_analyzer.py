import pandas as pd
import numpy as np
import re
import logging
from textblob import TextBlob
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

logger = logging.getLogger(__name__)

class SentimentAnalyzer:
    def __init__(self):
        self.lemmatizer = WordNetLemmatizer()
        try:
            self.stop_words = set(stopwords.words('english'))
        except:
            logger.warning("Stopwords not found. Please run: python -m nltk.downloader stopwords punkt wordnet")
            self.stop_words = set()
            
        self.relatable_keywords = [
            'relatable', 'so true', 'me', 'same', 'literally me', 'omg yes',
            'felt that', 'exactly', 'can relate', 'this is me', 'facts', 'fr'
        ]

    def preprocess_text(self, text: str) -> str:
        """Clean and preprocess text."""
        if not isinstance(text, str):
            return ""
            
        # Lowercase
        text = text.lower()
        # URL removal
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
        # Special character handling
        text = re.sub(r'\@\w+|\#|\W', ' ', text)
        
        try:
            # Tokenization
            tokens = word_tokenize(text)
            # Stopword processing & Lemmatization
            tokens = [self.lemmatizer.lemmatize(word) for word in tokens if word not in self.stop_words]
            return ' '.join(tokens)
        except Exception as e:
            # Fallback if NLTK data isn't loaded yet
            return text

    def analyze_sentiment(self, text: str) -> tuple:
        """Calculate sentiment using TextBlob."""
        if not text:
            return "Neutral", 0.0
            
        analysis = TextBlob(text)
        score = analysis.sentiment.polarity
        
        if score > 0.1:
            label = 'Positive'
        elif score < -0.1:
            label = 'Negative'
        else:
            label = 'Neutral'
            
        return label, score

    def analyze_relatability(self, text: str, original_text: str) -> str:
        """Determine if a comment is relatable based on keywords and NLP."""
        original_lower = original_text.lower()
        for keyword in self.relatable_keywords:
            if keyword in original_lower:
                return 'Relatable'
        
        # If highly emotional and positive/negative, might be relatable, but let's stick to heuristic
        return 'Neutral'

    def process_comments(self, df: pd.DataFrame) -> pd.DataFrame:
        """Process a dataframe of comments and add NLP predictions."""
        df = df.copy()
        
        if 'comment_text' not in df.columns:
            logger.error("Column 'comment_text' not found in dataframe.")
            return df
            
        logger.info("Preprocessing comments...")
        df['clean_text'] = df['comment_text'].apply(self.preprocess_text)
        
        logger.info("Analyzing sentiment and relatability...")
        # Apply sentiment
        sentiment_results = df['clean_text'].apply(self.analyze_sentiment)
        df['predicted_sentiment'] = [res[0] for res in sentiment_results]
        df['predicted_sentiment_score'] = [res[1] for res in sentiment_results]
        
        # Apply relatability using original text to catch exact phrases
        df['predicted_relatability'] = df['comment_text'].apply(lambda x: self.analyze_relatability("", str(x)))
        
        return df

    def evaluate_predictions(self, df: pd.DataFrame) -> dict:
        """Evaluate predictions against provided labels if available."""
        results = {}
        
        if 'sentiment_label' in df.columns and 'predicted_sentiment' in df.columns:
            y_true = df['sentiment_label']
            y_pred = df['predicted_sentiment']
            results['Sentiment'] = {
                'Accuracy': accuracy_score(y_true, y_pred),
                'F1 Score (weighted)': f1_score(y_true, y_pred, average='weighted', zero_division=0),
                'Confusion Matrix': confusion_matrix(y_true, y_pred).tolist()
            }
            
        if 'relatability_label' in df.columns and 'predicted_relatability' in df.columns:
            y_true = df['relatability_label']
            y_pred = df['predicted_relatability']
            results['Relatability'] = {
                'Accuracy': accuracy_score(y_true, y_pred),
                'F1 Score (weighted)': f1_score(y_true, y_pred, average='weighted', zero_division=0),
                'Confusion Matrix': confusion_matrix(y_true, y_pred).tolist()
            }
            
        return results

if __name__ == "__main__":
    from src.data_collection import DataCollector
    collector = DataCollector()
    comments_df = collector.load_dataset("comments_data.csv")
    analyzer = SentimentAnalyzer()
    processed_df = analyzer.process_comments(comments_df.head(100))
    eval_results = analyzer.evaluate_predictions(processed_df)
    print("Evaluation Results:", eval_results)
