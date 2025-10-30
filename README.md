# 📱 Analyzing Mental Health Trends on TikTok Using Topic Modeling and Sentiment Analysis

This project investigates mental health discussions on TikTok by applying natural language processing techniques to video descriptions, hashtags, audio transcripts, and user comments. Leveraging topic modeling and sentiment analysis, including toxicity evaluation, the study identifies prevalent mental health themes, gauges emotional tone, and provides insights into user interactions around mental health topics.

---

## 🎯 Objectives

- Identify and analyze major themes related to mental health on TikTok.
- Perform in-depth sentiment and toxicity analysis on user comments and video content.
- Provide quantitative and qualitative insights into mental health discourse on social media.

---

## 🔍 Methodology

### Data Collection & Preparation
- Collected data via TikTok API, focusing on mental health-related content.
- Converted raw JSON data into structured DataFrames, merging video descriptions, hashtags, and audio transcriptions for comprehensive analysis.

### Topic Modeling Techniques
- Used CountVectorizer + Latent Dirichlet Allocation (LDA) for initial topic extraction.
- Enhanced topic clarity by integrating domain-specific and social media-specific stop words.
- Experimented with TF-IDF vectorization + Non-negative Matrix Factorization (NMF).
- Explored transformer-based embeddings using BERTopic for deeper semantic extraction.
- Improved keyword distinctiveness with log odds method.

### Sentiment & Toxicity Analysis
- Conducted sentiment analysis using VADER Sentiment Analyzer.
- Evaluated toxicity using Detoxify, examining severe toxicity, insults, identity attacks, and threats.
- Visualized distributions and averages of sentiment and toxicity scores across topics using violin plots, histograms, and descriptive metrics.

### Validation & Qualitative Analysis
- Manually reviewed videos for topical relevance and contextual accuracy of automated analyses.

---

## 📈 Key Insights
- Distinctive topics emerged clearly after applying advanced NLP techniques.
- Toxicity levels and sentiment trends varied significantly across topics, providing critical insights for mental health professionals and content moderators.
- Interactive visualizations facilitated deeper understanding of user sentiments and discourse dynamics.

---

## 🛠️ Technologies Used
- Python (Pandas, NumPy, Matplotlib, Seaborn, scikit-learn)
- NLP: CountVectorizer, TF-IDF, LDA, NMF, BERTopic
- Sentiment Analysis: VADER
- Toxicity Analysis: Detoxify
- Visualization: Matplotlib, Seaborn, Plotly
- API & Data Extraction: TikTok API, Git repositories for comment extraction

---

## 🚀 Future Work
- Conduct longitudinal analysis to detect evolving mental health trends.
- Explore network analysis to understand user interactions and influence dynamics.
- Analyze the impact of significant external events (e.g., elections, pandemics) on mental health sentiment.

---

## 📄 Reports
(temporarily removed while we update the results/paper) check this again in a few weeks.
- 📊 Project Report
- 📑 Project Presentation 

---

## 📃 License
This project is intended for research and educational purposes only. Data used is publicly available or obtained via official APIs.
