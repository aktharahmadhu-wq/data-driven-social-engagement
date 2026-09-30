import pandas as pd
import numpy as np
import logging

logger = logging.getLogger(__name__)

class RecommendationEngine:
    def __init__(self):
        self.recommendations = []

    def analyze_category(self, df: pd.DataFrame, category: str, metric: str = 'engagement_rate') -> dict:
        """Analyze a category and return the best performing segment."""
        if category not in df.columns or metric not in df.columns:
            return None
            
        grouped = df.groupby(category)[metric].agg(['mean', 'count', 'std']).reset_index()
        # Filter out groups with too few samples
        min_samples = max(5, len(df) * 0.01)
        valid_groups = grouped[grouped['count'] >= min_samples]
        
        if len(valid_groups) == 0:
            return None
            
        best = valid_groups.loc[valid_groups['mean'].idxmax()]
        overall_mean = df[metric].mean()
        
        # Calculate a simple confidence score (0-100) based on sample size and difference from mean
        diff_from_mean_pct = (best['mean'] - overall_mean) / overall_mean
        z_score = (best['mean'] - overall_mean) / (best['std'] / np.sqrt(best['count'])) if best['std'] > 0 else 0
        
        confidence = min(100, max(0, int(stats.norm.cdf(z_score) * 100))) if not np.isnan(z_score) else 50
        
        return {
            'category': category,
            'recommendation': best[category],
            'metric': metric,
            'best_value': best['mean'],
            'overall_mean': overall_mean,
            'sample_size': best['count'],
            'confidence': confidence,
            'evidence': f"The '{best[category]}' segment achieved an average {metric} of {best['mean']:.4f} (across {int(best['count'])} samples), compared to the overall average of {overall_mean:.4f}."
        }

    def generate_recommendations(self, df: pd.DataFrame) -> list:
        """Generate full set of recommendations."""
        import scipy.stats as stats # imported here for the analyze_category z-score
        globals()['stats'] = stats
        
        self.recommendations = []
        
        categories = ['topic', 'format', 'posting_time', 'hook_type', 'day_of_week']
        metrics = ['engagement_rate', 'shares', 'viral_score']
        
        for cat in categories:
            for metric in metrics:
                if metric in df.columns and cat in df.columns:
                    rec = self.analyze_category(df, cat, metric)
                    if rec and rec['confidence'] > 75: # Only recommend if reasonably confident
                        self.recommendations.append({
                            'Type': f"Recommended {cat.title().replace('_', ' ')}",
                            'Recommendation': rec['recommendation'],
                            'Supporting Metric': metric,
                            'Evidence': rec['evidence'],
                            'Confidence': f"{rec['confidence']}%"
                        })
                        
        return self.recommendations

if __name__ == "__main__":
    from src.data_collection import DataCollector
    collector = DataCollector()
    df = collector.load_dataset("content_data.csv")
    engine = RecommendationEngine()
    recs = engine.generate_recommendations(df)
    for r in recs:
        print(f"{r['Type']}: {r['Recommendation']}\nEvidence: {r['Evidence']}\n")
