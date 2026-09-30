# Viva Questions & Answers

**1. What is the main objective of this project?**
To build a data science ecosystem that replaces guesswork with statistical evidence for social media content strategy, focusing on engagement and relatability.

**2. Which libraries did you use for data manipulation?**
Pandas for structured data manipulation and NumPy for numerical operations.

**3. What is the purpose of the Virality Prediction Engine?**
It uses machine learning to classify whether a piece of content will go viral based on non-leakage metadata features (like format, topic, and posting time).

**4. How do you handle missing values in your dataset?**
In `data_preprocessing.py`, numeric columns are filled with median values, and categorical text columns (like captions) are filled with empty strings to prevent errors.

**5. What is data leakage in machine learning?**
Data leakage occurs when information from outside the training dataset is used to create the model, or when the target variable is inadvertently included in the features. We avoided this by excluding 'shares' and 'saves' when predicting virality.

**6. Which NLP library did you use for sentiment analysis?**
We used TextBlob for polarity scoring and NLTK for text preprocessing (tokenization, stopwords, lemmatization).

**7. How do you quantify "Relatability"?**
For this demonstration, we used a heuristic approach combining keyword triggers (e.g., "literally me", "so true") and emotional sentiment.

**8. Why perform A/B testing?**
To determine if differences in engagement (e.g., between Short and Long formats) are statistically significant, rather than just due to random chance.

**9. What is a p-value?**
The probability of obtaining test results at least as extreme as the results actually observed, assuming the null hypothesis is true. A p-value < 0.05 indicates statistical significance.

**10. What does the Recommendation Engine do?**
It analyzes historical segments (e.g., topics or formats) and suggests the optimal content strategy backed by statistical confidence and sample size.

*(Additional 20 questions covering Overfitting, Streamlit, Database design, Future Scope, etc. would follow in a full viva session, focusing on the distinction between synthetic demo data and real-world API challenges).*
