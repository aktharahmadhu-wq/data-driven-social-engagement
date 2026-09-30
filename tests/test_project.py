import pytest
import pandas as pd
import numpy as np
import os
import sys

# Add src to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data_collection import DataCollector
from src.data_preprocessing import DataPreprocessor
from src.virality_model import ViralityModel
from src.sentiment_analyzer import SentimentAnalyzer
from src.ab_testing import ABTestingFramework
from src.recommendation_engine import RecommendationEngine
from src.trend_forecasting import TrendForecaster
from src.database import DatabaseLayer

@pytest.fixture
def sample_content():
    return pd.DataFrame({
        'content_id': ['C1', 'C2', 'C3'],
        'topic': ['Social Anxiety', 'Dating', 'Social Anxiety'],
        'format': ['Short', 'Long', 'Short'],
        'hook_type': ['Visual', 'Text', 'Visual'],
        'posting_time': ['Morning', 'Evening', 'Morning'],
        'engagement_rate': [5.5, 2.1, 6.0],
        'viral_class': ['Viral', 'Non-Viral', 'Viral'],
        'platform': ['Instagram', 'YouTube', 'Instagram'],
        'day_of_week': ['Monday', 'Tuesday', 'Wednesday'],
        'hour_of_day': [10, 18, 9],
        'video_length_sec': [15, 600, 30]
    })

@pytest.fixture
def sample_comments():
    return pd.DataFrame({
        'comment_text': ['This is so relatable!', 'I disagree.', 'Omg literally me'],
        'sentiment_label': ['Positive', 'Negative', 'Positive'],
        'relatability_label': ['Relatable', 'Neutral', 'Relatable']
    })

def test_data_preprocessing(sample_content):
    preprocessor = DataPreprocessor()
    clean_df = preprocessor.clean_content_data(sample_content)
    assert len(clean_df) == 3
    assert not clean_df.isnull().values.any()

def test_virality_model(sample_content):
    model = ViralityModel(models_dir="tests/models")
    results, best = model.train_and_evaluate(sample_content)
    # Even with small data, it should return results without crashing
    assert results is not None
    assert best is not None

def test_sentiment_analyzer(sample_comments):
    analyzer = SentimentAnalyzer()
    processed = analyzer.process_comments(sample_comments)
    assert 'predicted_sentiment' in processed.columns
    assert 'predicted_relatability' in processed.columns

def test_ab_testing():
    # Synthetic data for A/B testing
    df = pd.DataFrame({
        'format': ['Short']*50 + ['Long']*50,
        'views': np.random.normal(1000, 100, 100)
    })
    ab = ABTestingFramework()
    res = ab.analyze_experiment(df, 'format', 'Short', 'Long', ['views'])
    assert 'views' in res
    assert 'p_value' in res['views']

def test_recommendation_engine(sample_content):
    engine = RecommendationEngine()
    recs = engine.generate_recommendations(sample_content)
    # Small data might not generate recs due to min samples constraint
    assert isinstance(recs, list)

def test_trend_forecasting():
    df = pd.DataFrame({
        'date': pd.date_range(start='2023-01-01', periods=10, freq='W'),
        'topic': ['T1']*10,
        'engagement': np.arange(10)
    })
    forecaster = TrendForecaster()
    res = forecaster.analyze_trends(df)
    assert 'T1' in res
    assert 'forecast_values' in res['T1']
    assert len(res['T1']['forecast_values']) > 0
