import pandas as pd
import numpy as np
import logging
import os
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_auc_score

logger = logging.getLogger(__name__)

class ViralityModel:
    def __init__(self, models_dir: str = "models"):
        self.models_dir = models_dir
        os.makedirs(self.models_dir, exist_ok=True)
        self.model = None
        self.feature_names = None

    def prepare_data(self, df: pd.DataFrame):
        """Prepare features and target for the model."""
        df = df.copy()
        # Define target
        if 'viral_class' not in df.columns or df['viral_class'].nunique() < 2:
            # Create synthetic target if missing or only 1 class present
            logger.warning("'viral_class' missing or has < 2 unique classes. Generating based on engagement_rate.")
            threshold = df['engagement_rate'].quantile(0.75)
            df['viral_class'] = np.where(df['engagement_rate'] >= threshold, 'Viral', 'Non-Viral')
            
        y = (df['viral_class'] == 'Viral').astype(int)
        if df['viral_class'].nunique() < 2 or len(np.unique(y)) < 2:
             logger.warning("Still < 2 classes after synthetic generation. Forcing top 25% to be Viral.")
             df = df.sort_values('engagement_rate', ascending=False)
             viral_count = int(len(df) * 0.25)
             df['viral_class'] = 'Non-Viral'
             df.iloc[:viral_count, df.columns.get_loc('viral_class')] = 'Viral'
             y = (df['viral_class'] == 'Viral').astype(int)
        
        # Define features - avoiding data leakage (no engagement metrics)
        categorical_features = ['platform', 'content_type', 'topic', 'format', 'hook_type', 'day_of_week']
        numeric_features = ['hour_of_day', 'video_length_sec']
        
        # Ensure all columns exist
        cat_feats = [c for c in categorical_features if c in df.columns]
        num_feats = [c for c in numeric_features if c in df.columns]
        
        X = df[cat_feats + num_feats].copy()
        
        # Drop rows with missing target or features
        X = X.dropna()
        y = y.loc[X.index]
        
        return X, y, cat_feats, num_feats

    def train_and_evaluate(self, df: pd.DataFrame):
        """Train models, compare them, and save the best one."""
        X, y, cat_feats, num_feats = self.prepare_data(df)
        
        if len(X) == 0:
            logger.error("No valid data available for training.")
            return None
            
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', StandardScaler(), num_feats),
                ('cat', OneHotEncoder(handle_unknown='ignore'), cat_feats)
            ])
            
        models = {
            'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42, class_weight='balanced'),
            'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced'),
            'Gradient Boosting': GradientBoostingClassifier(n_estimators=100, random_state=42)
        }
        
        results = {}
        best_model_name = None
        best_f1 = -1
        best_pipeline = None
        
        for name, clf in models.items():
            pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                                     ('classifier', clf)])
            
            pipeline.fit(X_train, y_train)
            y_pred = pipeline.predict(X_test)
            
            if len(np.unique(y_test)) > 1:
                try:
                    y_prob = pipeline.predict_proba(X_test)[:, 1]
                    roc_auc = roc_auc_score(y_test, y_prob)
                except:
                    roc_auc = np.nan
            else:
                roc_auc = np.nan
                
            metrics = {
                'Accuracy': accuracy_score(y_test, y_pred),
                'Precision': precision_score(y_test, y_pred, zero_division=0),
                'Recall': recall_score(y_test, y_pred, zero_division=0),
                'F1 Score': f1_score(y_test, y_pred, zero_division=0),
                'ROC AUC': roc_auc,
                'Confusion Matrix': confusion_matrix(y_test, y_pred).tolist()
            }
            results[name] = metrics
            
            if metrics['F1 Score'] > best_f1:
                best_f1 = metrics['F1 Score']
                best_model_name = name
                best_pipeline = pipeline
                
        logger.info(f"Best model: {best_model_name} with F1 Score: {best_f1:.4f}")
        
        # Save best model
        self.model = best_pipeline
        joblib.dump(self.model, os.path.join(self.models_dir, 'virality_model.pkl'))
        
        return results, best_model_name

    def predict(self, df: pd.DataFrame):
        """Make predictions on new data."""
        if self.model is None:
            model_path = os.path.join(self.models_dir, 'virality_model.pkl')
            if os.path.exists(model_path):
                self.model = joblib.load(model_path)
            else:
                logger.error("Model not trained or saved yet.")
                return None
                
        X, _, _, _ = self.prepare_data(df)
        predictions = self.model.predict(X)
        probabilities = self.model.predict_proba(X)[:, 1]
        
        return predictions, probabilities

if __name__ == "__main__":
    from src.data_collection import DataCollector
    collector = DataCollector()
    content_df = collector.load_dataset("content_data.csv")
    model = ViralityModel()
    results, best = model.train_and_evaluate(content_df)
    print(f"Evaluation Results:\n{results}")
