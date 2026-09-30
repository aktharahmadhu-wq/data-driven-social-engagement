import sqlite3
import pandas as pd
import os
import logging

logger = logging.getLogger(__name__)

class DatabaseLayer:
    def __init__(self, db_path="data-driven-social.db"):
        """Initialize database connection"""
        self.db_path = db_path
        self._init_db()

    def get_connection(self):
        """Get database connection"""
        return sqlite3.connect(self.db_path)

    def _init_db(self):
        """Create necessary tables"""
        conn = self.get_connection()
        cursor = conn.cursor()

        # Content Table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS content (
                content_id TEXT PRIMARY KEY,
                date DATE,
                platform TEXT,
                content_type TEXT,
                topic TEXT,
                caption TEXT,
                hashtags TEXT,
                day_of_week TEXT,
                hour_of_day INTEGER,
                video_length_sec INTEGER,
                follower_count INTEGER,
                reach INTEGER,
                views INTEGER,
                likes INTEGER,
                comments INTEGER,
                shares INTEGER,
                saves INTEGER,
                retention_rate REAL,
                followers_gained INTEGER,
                engagement_rate REAL,
                viral_score REAL,
                viral_class TEXT,
                format TEXT,
                hook_type TEXT,
                posting_time TEXT
            )
        ''')

        # Comments Table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS comments (
                comment_id TEXT PRIMARY KEY,
                content_id TEXT,
                date DATE,
                comment_text TEXT,
                sentiment_label TEXT,
                sentiment_score REAL,
                relatability_label TEXT,
                topic TEXT,
                platform TEXT
            )
        ''')
        
        # Additional required tables
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS engagement (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                content_id TEXT,
                date DATE,
                likes INTEGER,
                comments INTEGER,
                shares INTEGER,
                saves INTEGER,
                views INTEGER,
                engagement_rate REAL
            )
        ''')

        cursor.execute('''
            CREATE TABLE IF NOT EXISTS sentiment (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                content_id TEXT,
                average_sentiment REAL,
                positive_ratio REAL,
                negative_ratio REAL,
                neutral_ratio REAL,
                relatability_score REAL
            )
        ''')

        cursor.execute('''
            CREATE TABLE IF NOT EXISTS predictions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                content_id TEXT,
                predicted_viral_score REAL,
                prediction_date DATE,
                model_used TEXT
            )
        ''')

        cursor.execute('''
            CREATE TABLE IF NOT EXISTS experiments (
                experiment_id TEXT PRIMARY KEY,
                name TEXT,
                start_date DATE,
                end_date DATE,
                variant_a TEXT,
                variant_b TEXT,
                metric TEXT,
                result TEXT
            )
        ''')

        conn.commit()
        conn.close()

    def import_dataframe(self, df: pd.DataFrame, table_name: str, if_exists: str = 'replace'):
        """Import a pandas DataFrame into a database table"""
        conn = self.get_connection()
        df.to_sql(table_name, conn, if_exists=if_exists, index=False)
        conn.close()

    def fetch_data(self, query: str, params: tuple = ()) -> pd.DataFrame:
        """Fetch data from database as pandas DataFrame"""
        conn = self.get_connection()
        df = pd.read_sql_query(query, conn, params=params)
        conn.close()
        return df

    def execute_query(self, query: str, params: tuple = ()):
        """Execute a single query"""
        conn = self.get_connection()
        cursor = conn.cursor()
        cursor.execute(query, params)
        conn.commit()
        conn.close()
