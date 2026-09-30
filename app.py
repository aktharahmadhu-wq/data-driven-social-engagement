import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import os
import sys

# Add src to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.data_collection import DataCollector
from src.data_preprocessing import DataPreprocessor
from src.virality_model import ViralityModel
from src.sentiment_analyzer import SentimentAnalyzer
from src.ab_testing import ABTestingFramework
from src.recommendation_engine import RecommendationEngine
from src.trend_forecasting import TrendForecaster
from src.database import DatabaseLayer

# Page config
st.set_page_config(page_title="Data-Driven Social Engagement", page_icon="📊", layout="wide")

# Custom CSS for premium UI
st.markdown("""
<style>
    .main-header { font-size: 2.5rem; font-weight: 700; color: #1E3A8A; margin-bottom: 0; }
    .sub-header { font-size: 1.2rem; color: #64748B; margin-bottom: 2rem; }
    .metric-card { background-color: #F8FAFC; border-radius: 10px; padding: 1.5rem; box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1); }
    .stProgress .st-bo { background-color: #3B82F6; }
    .disclaimer { font-size: 0.8rem; color: #94A3B8; font-style: italic; }
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_data():
    """Load all necessary datasets."""
    try:
        collector = DataCollector()
        content_df = collector.load_dataset("content_data.csv")
        comments_df = collector.load_dataset("comments_data.csv")
        ab_test_df = collector.load_dataset("ab_testing_data.csv")
        trend_df = collector.load_dataset("trend_data.csv")
        content_series = collector.load_dataset("content_series.csv")
        return content_df, comments_df, ab_test_df, trend_df, content_series
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None, None, None, None, None

def main():
    st.sidebar.title("Navigation")
    pages = [
        "Executive Dashboard", 
        "Content Performance", 
        "Virality Prediction", 
        "Audience Sentiment", 
        "A/B Testing", 
        "Strategy Recommendations", 
        "Trend Forecasting",
        "Data Upload & Config"
    ]
    selection = st.sidebar.radio("Go to", pages)

    st.sidebar.markdown("---")
    st.sidebar.markdown("<p class='disclaimer'>Note: The demonstration dataset used in this project is synthetic and is intended for academic development, testing and demonstration. It does not represent real Instagram, YouTube, or individual user data.</p>", unsafe_allow_html=True)

    content_df, comments_df, ab_test_df, trend_df, content_series = load_data()
    
    if content_df is None:
        st.warning("Please upload datasets in the 'Data Upload & Config' tab or ensure the default datasets are present in data/raw/.")
        return

    if selection == "Executive Dashboard":
        st.markdown("<h1 class='main-header'>Executive Dashboard</h1>", unsafe_allow_html=True)
        st.markdown("<p class='sub-header'>High-level overview of social media performance and engagement.</p>", unsafe_allow_html=True)
        
        # Top level metrics
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Total Content", len(content_df))
        col2.metric("Total Views", f"{content_df['views'].sum():,}")
        col3.metric("Followers Gained", f"{content_df['followers_gained'].sum():,}")
        col4.metric("Avg Engagement Rate", f"{content_df['engagement_rate'].mean():.2f}%")
        
        st.markdown("### Engagement Overview")
        col_chart1, col_chart2 = st.columns(2)
        
        with col_chart1:
            fig1 = px.pie(content_df, names='platform', title='Content Distribution by Platform', hole=0.4, color_discrete_sequence=px.colors.qualitative.Pastel)
            st.plotly_chart(fig1, use_container_width=True)
            
        with col_chart2:
            fig2 = px.bar(content_df.groupby('topic')['views'].mean().reset_index(), x='topic', y='views', title='Average Views by Topic', color='views', color_continuous_scale='Blues')
            st.plotly_chart(fig2, use_container_width=True)
            
    elif selection == "Content Performance":
        st.markdown("<h1 class='main-header'>Content Performance Tracker</h1>", unsafe_allow_html=True)
        
        st.dataframe(content_df.head(10))
        
        col1, col2 = st.columns(2)
        with col1:
            fig = px.scatter(content_df, x='views', y='engagement_rate', color='topic', size='shares', hover_name='content_id', title='Views vs Engagement Rate (Bubble size = Shares)')
            st.plotly_chart(fig, use_container_width=True)
            
        with col2:
            fig2 = px.box(content_df, x='format', y='retention_rate', color='platform', title='Retention Rate by Format')
            st.plotly_chart(fig2, use_container_width=True)

    elif selection == "Virality Prediction":
        st.markdown("<h1 class='main-header'>Virality Prediction Engine</h1>", unsafe_allow_html=True)
        
        if st.button("Train Virality Model"):
            with st.spinner("Training models..."):
                model = ViralityModel()
                results, best = model.train_and_evaluate(content_df)
                st.success(f"Model trained successfully! Best model: {best}")
                
                st.write("### Model Metrics")
                st.json(results)
                
        # Show distribution
        if 'viral_class' in content_df.columns:
            fig = px.histogram(content_df, x='viral_class', color='topic', title='Virality Distribution across Topics')
            st.plotly_chart(fig, use_container_width=True)

    elif selection == "Audience Sentiment":
        st.markdown("<h1 class='main-header'>Audience Sentiment Analyzer</h1>", unsafe_allow_html=True)
        
        if st.button("Run NLP Analysis on Comments"):
            with st.spinner("Processing text..."):
                analyzer = SentimentAnalyzer()
                processed = analyzer.process_comments(comments_df.head(500)) # limit for demo speed
                st.session_state['processed_comments'] = processed
                
        if 'processed_comments' in st.session_state:
            df_nlp = st.session_state['processed_comments']
            
            col1, col2 = st.columns(2)
            with col1:
                fig = px.pie(df_nlp, names='predicted_sentiment', title='Predicted Sentiment Distribution')
                st.plotly_chart(fig, use_container_width=True)
            with col2:
                fig2 = px.pie(df_nlp, names='predicted_relatability', title='Relatability Detection')
                st.plotly_chart(fig2, use_container_width=True)
                
            st.dataframe(df_nlp[['comment_text', 'predicted_sentiment', 'predicted_relatability']].head())

    elif selection == "A/B Testing":
        st.markdown("<h1 class='main-header'>A/B Testing Framework</h1>", unsafe_allow_html=True)
        
        ab = ABTestingFramework()
        results = ab.run_all_tests(ab_test_df)
        
        for test_name, metrics in results.items():
            with st.expander(f"Experiment: {test_name}", expanded=True):
                for metric, res in metrics.items():
                    if 'error' in res:
                        continue
                    
                    st.write(f"**Metric:** {metric}")
                    col1, col2, col3, col4 = st.columns(4)
                    col1.metric("Variant A Mean", f"{res['mean_a']:.2f}")
                    col2.metric("Variant B Mean", f"{res['mean_b']:.2f}")
                    col3.metric("Difference (%)", f"{res['percentage_difference']:.2f}%")
                    
                    sig_color = "green" if res['is_significant'] else "gray"
                    sig_text = "Significant" if res['is_significant'] else "Not Significant"
                    col4.markdown(f"<div style='color:{sig_color}; font-weight:bold;'>{sig_text} (p={res['p_value']:.4f})</div>", unsafe_allow_html=True)
                    st.divider()

    elif selection == "Strategy Recommendations":
        st.markdown("<h1 class='main-header'>Engagement Optimization Recommender</h1>", unsafe_allow_html=True)
        
        engine = RecommendationEngine()
        recs = engine.generate_recommendations(content_df)
        
        for r in recs:
            st.info(f"**{r['Type']}:** {r['Recommendation']}\n\n*Confidence:* {r['Confidence']}\n\n*Evidence:* {r['Evidence']}")

        st.markdown("### Planned Content Series (Demonstration)")
        st.dataframe(content_series)

    elif selection == "Trend Forecasting":
        st.markdown("<h1 class='main-header'>Trend Forecasting Module</h1>", unsafe_allow_html=True)
        st.write("Predicting rising relatable struggles based on historical engagement.")
        
        forecaster = TrendForecaster()
        forecasts = forecaster.analyze_trends(trend_df)
        
        selected_topic = st.selectbox("Select Topic to Forecast", list(forecasts.keys()))
        if selected_topic:
            data = forecasts[selected_topic]
            
            # Create chart
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=data['historical_dates'], y=data['historical_values'], mode='lines+markers', name='Historical'))
            fig.add_trace(go.Scatter(x=data['forecast_dates'], y=data['forecast_values'], mode='lines+markers', line=dict(dash='dash'), name='Forecast'))
            
            fig.update_layout(title=f"Trend Forecast: {selected_topic}", xaxis_title="Date", yaxis_title="Engagement")
            st.plotly_chart(fig, use_container_width=True)
            
            col1, col2 = st.columns(2)
            col1.metric("Historical Growth Rate", f"{data['growth_rate']:.1f}%")
            col2.metric("Trend Direction", data['trend_direction'])
            
            st.caption("Synthetic Demonstration Result")

    elif selection == "Data Upload & Config":
        st.markdown("<h1 class='main-header'>Data Upload & Configuration</h1>", unsafe_allow_html=True)
        
        uploaded_file = st.file_uploader("Upload new content dataset (CSV)", type="csv")
        if uploaded_file is not None:
            new_df = pd.read_csv(uploaded_file)
            st.success("File uploaded successfully!")
            st.dataframe(new_df.head())
            
        if st.button("Initialize Local Database (SQLite)"):
            try:
                db = DatabaseLayer()
                db.import_dataframe(content_df, 'content')
                db.import_dataframe(comments_df, 'comments')
                st.success("Database initialized and datasets imported!")
            except Exception as e:
                st.error(f"Database error: {e}")

if __name__ == "__main__":
    main()
