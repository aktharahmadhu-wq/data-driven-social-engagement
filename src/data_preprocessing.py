import pandas as pd
import numpy as np
import os
import logging
from src.data_collection import DataCollector

logger = logging.getLogger(__name__)

class DataPreprocessor:
    def __init__(self, data_dir: str = "data"):
        self.data_dir = data_dir
        self.processed_dir = os.path.join(data_dir, "processed")
        os.makedirs(self.processed_dir, exist_ok=True)
        
    def clean_content_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean and preprocess content data"""
        df = df.copy()
        
        # Remove duplicates
        df = df.drop_duplicates()
        
        # Handle missing values
        if 'caption' in df.columns:
            df['caption'] = df['caption'].fillna('')
        if 'hashtags' in df.columns:
            df['hashtags'] = df['hashtags'].fillna('')
            
        # Numerical columns fill with median or 0
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            df[col] = df[col].fillna(df[col].median())
            
        # Format dates
        if 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'], errors='coerce')
            df['date'] = df['date'].dt.date
            
        return df

    def clean_comments_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean and preprocess comments data"""
        df = df.copy()
        
        # Remove duplicates
        df = df.drop_duplicates()
        
        # Drop rows with no comment text
        if 'comment_text' in df.columns:
            df = df.dropna(subset=['comment_text'])
            df['comment_text'] = df['comment_text'].astype(str)
            
        return df

    def run_pipeline(self) -> tuple:
        """Run the preprocessing pipeline"""
        collector = DataCollector(self.data_dir)
        try:
            content_df, comments_df = collector.ingest_data()
        except FileNotFoundError:
            logger.error("Could not ingest data. Ensure raw files are present.")
            return None, None
            
        clean_content = self.clean_content_data(content_df)
        clean_comments = self.clean_comments_data(comments_df)
        
        # Save processed data
        clean_content.to_csv(os.path.join(self.processed_dir, "content_data_clean.csv"), index=False)
        clean_comments.to_csv(os.path.join(self.processed_dir, "comments_data_clean.csv"), index=False)
        
        logger.info("Preprocessing complete. Saved clean datasets to processed directory.")
        return clean_content, clean_comments

if __name__ == "__main__":
    preprocessor = DataPreprocessor()
    preprocessor.run_pipeline()
