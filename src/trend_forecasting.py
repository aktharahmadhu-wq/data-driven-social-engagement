import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import logging

logger = logging.getLogger(__name__)

class TrendForecaster:
    def __init__(self, forecast_periods: int = 4):
        self.forecast_periods = forecast_periods
        self.forecasts = {}

    def analyze_trends(self, df: pd.DataFrame, date_col: str = 'week', metric_col: str = 'engagement_events', group_col: str = 'topic') -> dict:
        """Analyze historical trends and forecast future values."""
        df = df.copy()
        df[date_col] = pd.to_datetime(df[date_col])
        
        # Aggregate by group and date (e.g. weekly or monthly)
        # Assuming the data is already aggregated by some frequency if we just use it directly,
        # but let's aggregate by week just in case.
        df['period'] = df[date_col].dt.to_period('W').dt.start_time
        
        grouped = df.groupby([group_col, 'period'])[metric_col].sum().reset_index()
        
        results = {}
        
        for topic in grouped[group_col].unique():
            topic_data = grouped[grouped[group_col] == topic].sort_values('period')
            if len(topic_data) < 3:
                continue # Not enough data
                
            # Prepare data for simple linear regression
            y = topic_data[metric_col].values
            X = np.arange(len(y)).reshape(-1, 1)
            
            model = LinearRegression()
            model.fit(X, y)
            
            # Forecast
            future_X = np.arange(len(y), len(y) + self.forecast_periods).reshape(-1, 1)
            forecast_y = model.predict(future_X)
            
            # Calculate growth rate
            historical_growth = (y[-1] - y[0]) / max(y[0], 1) * 100
            
            # Create forecast dates
            last_date = topic_data['period'].iloc[-1]
            future_dates = [last_date + pd.Timedelta(weeks=i+1) for i in range(self.forecast_periods)]
            
            results[topic] = {
                'historical_dates': topic_data['period'].tolist(),
                'historical_values': y.tolist(),
                'forecast_dates': future_dates,
                'forecast_values': forecast_y.tolist(),
                'growth_rate': historical_growth,
                'trend_direction': 'Upward' if model.coef_[0] > 0 else 'Downward',
                'is_synthetic': True # Label as synthetic demo
            }
            
        self.forecasts = results
        return results

if __name__ == "__main__":
    from src.data_collection import DataCollector
    collector = DataCollector()
    trend_df = collector.load_dataset("trend_data.csv")
    forecaster = TrendForecaster()
    res = forecaster.analyze_trends(trend_df)
    print(f"Generated forecasts for {len(res)} topics.")
