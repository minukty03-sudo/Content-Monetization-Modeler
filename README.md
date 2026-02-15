# 💰 Content Monetization Modeler

A **machine learning-powered Streamlit application** that predicts YouTube ad revenue based on video performance metrics, engagement data, and metadata.

---

## 📋 Table of Contents

1. [Project Overview](#project-overview)
2. [Dataset Description](#dataset-description)
3. [Preprocessing Pipeline](#preprocessing-pipeline)
4. [Feature Engineering](#feature-engineering)
5. [Model Building & Comparison](#model-building--comparison)
6. [Model Evaluation Results](#model-evaluation-results)
7. [Application Features](#application-features)
8. [Installation & Setup](#installation--setup)
9. [Usage](#usage)
10. [File Structure](#file-structure)
11. [Key Findings](#key-findings)

---

## 📖 Project Overview

This project develops a **Gradient Boosting Regressor** model to predict YouTube ad revenue (`ad_revenue_usd`) based on:
- Video performance metrics (views, likes, comments, watch time)
- Channel statistics (subscribers)
- Metadata (content category, target device, country)

The model is deployed via a **Streamlit web application** with interactive prediction capabilities and data visualization dashboards.

---

## 📊 Dataset Description

| Attribute | Details |
|-----------|---------|
| **Source** | `youtube_ad_revenue_dataset.csv` |
| **Original Rows** | 122,400 |
| **Rows After Cleaning** | 120,000 |
| **Features** | 12 original + engineered features |

### Original Features

| Column | Type | Description |
|--------|------|-------------|
| `video_id` | String | Unique video identifier |
| `date` | DateTime | Video upload date |
| `views` | Integer | Total view count |
| `likes` | Float | Total likes (may have missing values) |
| `comments` | Float | Total comments (may have missing values) |
| `watch_time_minutes` | Float | Total watch time in minutes |
| `video_length_minutes` | Float | Video duration |
| `subscribers` | Integer | Channel subscriber count |
| `category` | String | Content category |
| `device` | String | Primary viewing device |
| `country` | String | Target country |
| `ad_revenue_usd` | Float | **Target variable** - Ad revenue in USD |

### Missing Values (Before Cleaning)

| Column | Missing Count |
|--------|---------------|
| `likes` | 6,117 |
| `comments` | 6,112 |
| `watch_time_minutes` | 6,105 |

---

## 🔧 Preprocessing Pipeline

### 1. Remove Duplicates

```python
# Check for duplicates
df.duplicated().sum()  # Found: 2,400 duplicates

# Remove duplicates
df.drop_duplicates(inplace=True)
# Final rows: 120,000
```

**Rationale**: Duplicate rows can bias the model by over-representing certain data points.

---

### 2. Handle Missing Values

```python
# Strategy 1: Fill likes & comments with 0
df[['likes', 'comments']] = df[['likes', 'comments']].fillna(0)

# Strategy 2: Fill watch_time with median
df['watch_time_minutes'] = df['watch_time_minutes'].fillna(
    df['watch_time_minutes'].median()
)
```

| Column | Strategy | Reason |
|--------|----------|--------|
| `likes` | Fill with 0 | No likes recorded = 0 likes |
| `comments` | Fill with 0 | No comments recorded = 0 comments |
| `watch_time_minutes` | Fill with median | Median is robust to outliers |

---

### 3. Encode Categorical Variables

```python
# One-Hot Encoding with drop_first=True
df = pd.get_dummies(
    df, 
    columns=['category', 'device', 'country'], 
    drop_first=True, 
    dtype=int
)
```

**Encoded Columns Created:**
- `category_Entertainment`, `category_Gaming`, `category_Lifestyle`, `category_Music`, `category_Tech`
- `device_Mobile`, `device_TV`, `device_Tablet`
- `country_CA`, `country_DE`, `country_IN`, `country_UK`, `country_US`

**Note**: `drop_first=True` prevents multicollinearity by using reference categories:
- Category reference: `Education`
- Device reference: `Desktop`
- Country reference: `AU`

---

## ⚙️ Feature Engineering

### Engagement Rate

A new feature was engineered to capture user engagement:

```python
df['engagement_rate'] = (df['likes'] + df['comments']) / df['views']
df['engagement_rate'] = df['engagement_rate'].fillna(0).round(4)
```

**Formula**: `engagement_rate = (likes + comments) / views`

This metric captures how actively viewers interact with the content relative to total views.

---

### Date Feature Extraction

```python
df['Date'] = pd.to_datetime(df['date'], errors='coerce')
df['Year'] = df['Date'].dt.year
df['Quarter'] = 'Q' + df['Date'].dt.quarter.astype(str)
```

---

## 🤖 Model Building & Comparison

Five regression models were evaluated:

```python
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor

models = {
    "Linear": LinearRegression(),
    "Ridge": Ridge(),
    "Lasso": Lasso(),
    "RandomForest": RandomForestRegressor(random_state=42, n_jobs=-1),
    "GradientBoosting": GradientBoostingRegressor(random_state=42)
}
```

**Data Split**: 80% training / 20% testing

---

## 📈 Model Evaluation Results

| Model | R² Score | RMSE | MAE | MAPE |
|-------|----------|------|-----|------|
| Linear Regression | 0.949 | 14.02 | 4.79 | 2.06% |
| Ridge Regression | 0.949 | 14.02 | 4.79 | 2.06% |
| Lasso Regression | 0.949 | 14.02 | 4.79 | 2.06% |
| Random Forest | 0.949 | 13.91 | 3.55 | 1.52% |
| **Gradient Boosting** | **0.952** | **13.53** | **3.69** | **1.59%** |

### 🏆 Best Model: Gradient Boosting Regressor

**Why Gradient Boosting?**
- Highest R² score (0.952) - explains 95.2% of variance
- Lowest RMSE (13.53) - smallest prediction errors
- Low MAPE (1.59%) - excellent percentage accuracy

---

## 🌐 Application Features

The Streamlit app (`app.py`) provides four main sections:

### 1. Home Page
- Project introduction
- App capabilities overview

### 2. Prediction Page
- Interactive input form for video metrics
- Real-time engagement rate calculation
- Revenue prediction with model inference

### 3. Data Insights
- Revenue vs. Views scatter plot
- Revenue by Category bar chart
- Revenue Distribution histogram

### 4. Model Insights
- Feature importance visualization
- Top contributing factors
- Interpretation guidelines

---

## 🚀 Installation & Setup

### Prerequisites
- Python 3.8+
- pip package manager

### Installation Steps

```bash
# 1. Clone/Navigate to project directory
cd "Content Monetization Modeler"

# 2. Create virtual environment
python -m venv venv

# 3. Activate virtual environment
# Windows:
venv\Scripts\activate
# macOS/Linux:
source venv/bin/activate

# 4. Install dependencies
pip install streamlit pandas numpy matplotlib seaborn scikit-learn

# 5. Run the application
streamlit run app.py
```

---

## 💻 Usage

### Making Predictions

1. Launch the app: `streamlit run app.py`
2. Navigate to **Prediction** page
3. Enter video metrics:
   - Views, Likes, Comments
   - Watch Time, Video Length
   - Subscribers, Year
   - Category, Device, Country
4. Click **Calculate Revenue**
5. View predicted ad revenue

### Example Input

| Metric | Value |
|--------|-------|
| Views | 100,000 |
| Likes | 5,000 |
| Comments | 500 |
| Watch Time (min) | 50,000 |
| Video Length (min) | 15 |
| Subscribers | 250,000 |
| Category | Gaming |
| Device | Mobile |
| Country | US |

---

## 📁 File Structure

```
Content Monetization Modeler/
│
├── app.py                          # Streamlit web application
├── data.ipynb                      # Data preprocessing & model training notebook
├── model.pkl                       # Trained Gradient Boosting model
├── youtube_ad_revenue_dataset.csv  # Raw dataset
├── README.md                       # This documentation file
└── venv/                           # Python virtual environment
```

---

## 🔍 Key Findings

### Feature Importance (Top Factors)

1. **watch_time_minutes** - Most influential predictor
2. **likes** - Strong engagement signal
3. **subscribers** - Channel authority indicator
4. **engagement_rate** - Engineered interaction metric

### Correlation Insights

- `watch_time_minutes` has the highest correlation with `ad_revenue_usd` (0.989)
- `likes` shows moderate positive correlation (0.146)
- `video_length_minutes` has minimal impact on revenue

### Outlier Analysis

Using IQR and Z-Score methods, **0 outliers** were detected in the `ad_revenue_usd` column, indicating clean target data.

---

## 📝 Future Improvements

- [ ] Add time-series forecasting capabilities
- [ ] Implement model retraining pipeline
- [ ] Add more granular category sub-classifications
- [ ] Deploy to cloud platform (Streamlit Cloud / Heroku)

---

## 👤 Author

Created for **Content Monetization Project** - GUVI

---

*Last Updated: January 2026*
