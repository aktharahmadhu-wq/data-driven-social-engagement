import pandas as pd
import os
import logging
from typing import Tuple, Dict, Any

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class DataCollector:
    def __init__(self, data_dir: str = "data"):
        self.data_dir = data_dir
        self.raw_dir = os.path.join(data_dir, "raw")
        self.processed_dir = os.path.join(data_dir, "processed")
        os.makedirs(self.raw_dir, exist_ok=True)
        os.makedirs(self.processed_dir, exist_ok=True)

    def load_dataset(self, filename: str) -> pd.DataFrame:
        """Loads a dataset from the raw directory."""
        filepath = os.path.join(self.raw_dir, filename)
        if not os.path.exists(filepath):
            logger.error(f"File not found: {filepath}")
            raise FileNotFoundError(f"Required file {filepath} is missing.")
            
        if filename.endswith(".csv"):
            df = pd.read_csv(filepath)
        elif filename.endswith((".xls", ".xlsx")):
            df = pd.read_excel(filepath)
        else:
            raise ValueError("Unsupported file format. Please provide .csv or .xlsx")
            
        logger.info(f"Loaded {filename} with shape {df.shape}")
        return df

    def validate_data(self, df: pd.DataFrame, expected_columns: list = None) -> Dict[str, Any]:
        """Validates the loaded dataframe and returns a summary report."""
        report = {
            "rows": len(df),
            "columns": len(df.columns),
            "missing_values": df.isnull().sum().to_dict(),
            "duplicates": df.duplicated().sum(),
            "data_types": df.dtypes.astype(str).to_dict()
        }
        
        if expected_columns:
            missing_cols = [col for col in expected_columns if col not in df.columns]
            if missing_cols:
                logger.warning(f"Missing expected columns: {missing_cols}")
                report["missing_expected_columns"] = missing_cols
                
        logger.info("Data validation completed.")
        return report

    def ingest_data(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Ingest all required datasets for the pipeline."""
        content_df = self.load_dataset("content_data.csv")
        comments_df = self.load_dataset("comments_data.csv")
        
        # Log validation report
        logger.info(f"Content Data Validation: {self.validate_data(content_df)}")
        logger.info(f"Comments Data Validation: {self.validate_data(comments_df)}")
        
        return content_df, comments_df

if __name__ == "__main__":
    collector = DataCollector()
    try:
        content, comments = collector.ingest_data()
        print("Data collection successful.")
    except Exception as e:
        print(f"Data collection failed: {e}")
