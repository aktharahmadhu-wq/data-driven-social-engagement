import pandas as pd
import numpy as np
from scipy import stats
import logging

logger = logging.getLogger(__name__)

class ABTestingFramework:
    def __init__(self, confidence_level: float = 0.95):
        self.alpha = 1 - confidence_level

    def run_two_sample_test(self, group_a: pd.Series, group_b: pd.Series, metric_name: str) -> dict:
        """Run a statistical test comparing two groups."""
        # Drop NaNs
        a = group_a.dropna()
        b = group_b.dropna()
        
        if len(a) == 0 or len(b) == 0:
            return {"error": "Insufficient data"}

        mean_a, mean_b = a.mean(), b.mean()
        median_a, median_b = a.median(), b.median()
        diff = mean_b - mean_a
        pct_diff = (diff / mean_a) * 100 if mean_a != 0 else 0
        
        # Check normality to decide test (Shapiro-Wilk)
        # For simplicity in this demonstration, we'll use t-test for large samples
        # and Mann-Whitney for non-normal if required, but let's default to T-Test (Welch's)
        t_stat, p_value = stats.ttest_ind(a, b, equal_var=False)
        
        is_significant = p_value < self.alpha
        
        # Calculate confidence interval for the difference of means
        se_a, se_b = a.std() / np.sqrt(len(a)), b.std() / np.sqrt(len(b))
        se_diff = np.sqrt(se_a**2 + se_b**2)
        margin_of_error = stats.t.ppf(1 - self.alpha/2, df=min(len(a)-1, len(b)-1)) * se_diff
        ci_lower = diff - margin_of_error
        ci_upper = diff + margin_of_error

        return {
            "metric": metric_name,
            "mean_a": mean_a,
            "mean_b": mean_b,
            "median_a": median_a,
            "median_b": median_b,
            "difference": diff,
            "percentage_difference": pct_diff,
            "p_value": p_value,
            "is_significant": is_significant,
            "ci_lower": ci_lower,
            "ci_upper": ci_upper,
            "test_used": "Welch's t-test"
        }

    def analyze_experiment(self, df: pd.DataFrame, factor_col: str, group_a_val: str, group_b_val: str, metric_cols: list) -> dict:
        """Analyze an A/B test between two values of a factor."""
        results = {}
        
        group_a = df[df[factor_col] == group_a_val]
        group_b = df[df[factor_col] == group_b_val]
        
        for metric in metric_cols:
            if metric in df.columns:
                res = self.run_two_sample_test(group_a[metric], group_b[metric], metric)
                results[metric] = res
                
        return results

    def run_all_tests(self, df: pd.DataFrame) -> dict:
        """Run all predefined tests based on the project requirements."""
        metrics_to_test = ['views', 'engagement_rate', 'shares', 'saves', 'retention_rate']
        all_results = {}
        
        # Format: Short vs Long
        if 'format' in df.columns:
            all_results['Format (Long vs Short)'] = self.analyze_experiment(
                df, 'format', 'Short', 'Long', metrics_to_test
            )
            
        # Hook: Visual vs Text
        if 'hook_type' in df.columns:
            all_results['Hook (Text vs Visual)'] = self.analyze_experiment(
                df, 'hook_type', 'Visual', 'Text', metrics_to_test
            )
            
        # Posting Time: (Example Morning vs Evening)
        if 'posting_time' in df.columns:
            all_results['Time (Morning vs Evening)'] = self.analyze_experiment(
                df, 'posting_time', 'Morning', 'Evening', metrics_to_test
            )
            
        return all_results

if __name__ == "__main__":
    from src.data_collection import DataCollector
    collector = DataCollector()
    ab_df = collector.load_dataset("ab_testing_data.csv")
    ab = ABTestingFramework()
    results = ab.run_all_tests(ab_df)
    print("A/B Testing completed.")
