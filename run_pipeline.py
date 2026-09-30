import os
import sys
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def main():
    logger.info("Starting Data-Driven Social Engagement Pipeline...")
    
    # 1. Preprocessing
    from src.data_preprocessing import DataPreprocessor
    preprocessor = DataPreprocessor()
    content_clean, comments_clean = preprocessor.run_pipeline()
    
    if content_clean is None:
        logger.error("Pipeline failed at preprocessing.")
        return
        
    # 2. NLP Sentiment Analysis
    from src.sentiment_analyzer import SentimentAnalyzer
    analyzer = SentimentAnalyzer()
    processed_comments = analyzer.process_comments(comments_clean)
    logger.info("Sentiment Analysis completed.")
    
    # 3. Virality Model Training
    from src.virality_model import ViralityModel
    model = ViralityModel()
    results, best = model.train_and_evaluate(content_clean)
    logger.info(f"Virality Model trained. Best: {best}")
    
    # 4. Recommendation Engine
    from src.recommendation_engine import RecommendationEngine
    engine = RecommendationEngine()
    recs = engine.generate_recommendations(content_clean)
    logger.info(f"Generated {len(recs)} recommendations.")
    
    # 5. Database Initialization
    from src.database import DatabaseLayer
    db = DatabaseLayer()
    db.import_dataframe(content_clean, 'content')
    db.import_dataframe(processed_comments, 'comments')
    logger.info("Database initialized and data imported.")
    
    logger.info("Pipeline completed successfully!")

if __name__ == "__main__":
    main()
