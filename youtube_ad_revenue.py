import streamlit as st 
import pandas as pd 
import numpy as np 
import pickle 
import matplotlib.pyplot as plt 
import seaborn as sns 

# --------------------------------------------------
# Page Config
# --------------------------------------------------
st.set_page_config(
    page_title="Content Monetization Modeler",
    layout="wide",
    page_icon="💰"
)

# --------------------------------------------------
# Load Model
# --------------------------------------------------
with open("model.pkl", "rb") as f:
    model = pickle.load(f)

@st.cache_data
def load_data():
    try:
        df = pd.read_csv('youtube_ad_revenue_dataset.csv', nrows=5000)
        # Convert date to datetime
        if 'date' in df.columns:
            df['Date'] = pd.to_datetime(df['date'], errors='coerce')
        return df
    except:
        return pd.DataFrame()

df_sample = load_data()


# --------------------------------------------------
# Sidebar
# --------------------------------------------------
st.sidebar.image(
    "https://upload.wikimedia.org/wikipedia/commons/thumb/b/b8/YouTube_Logo_2017.svg/300px-YouTube_Logo_2017.svg.png",
    width=150
)
st.sidebar.title("Navigation")
page = st.sidebar.radio(
    "Go to",
    ["Home", "Prediction", "Data Insights", "Model Insights"]
)

# --------------------------------------------------
# HOME PAGE
# --------------------------------------------------
if page == "Home":
    st.title("💰 YouTube Content Monetization Modeler")

    st.markdown("""
    👋 **Welcome to the YouTube Ad Revenue Prediction App!**

    This app uses a **Gradient Boosting Regressor**
    to estimate **YouTube ad revenue** based on video performance
    and engagement metrics.

    ### 🔍 What can you do?
    - Predict ad revenue
    - Data insights
    - Model Insights""")

    st.image(
        "https://images.unsplash.com/photo-1611162617474-5b21e879e113",
        use_container_width=True
    )

    # Custom Styling
st.markdown("""
    <style>
    .main {
        background-color: #f5f7f9;
    }
    .stButton>button {
        background-color: #ff4b4b;
        color: white;
        border-radius: 10px;
        height: 3em;
        width: 100%;
        font-weight: bold;
    }
    </style>
    """, unsafe_allow_html=True)

# Mappings (Extracted from training data logic)
CATEGORIES = ["Education", "Entertainment", "Gaming", "Lifestyle", "Music", "Tech"]
DEVICES = ["Desktop", "Mobile", "TV", "Tablet"]
COUNTRIES = ["AU", "CA", "DE", "IN", "UK", "US"]

if page == "Prediction":
    st.title("🚀 Revenue Prediction")
    
    col1, col2 = st.columns([1, 1], gap="large")
    
    with col1:
        st.subheader("📈 Performance Metrics")
        views = st.number_input("Total Views", min_value=0, value=10000, step=100, help="Total expected views for the video.")
        likes = st.number_input("Total Likes", min_value=0, value=500, step=10)
        comments = st.number_input("Total Comments", min_value=0, value=50, step=5)
        watch_time = st.number_input("Watch Time (Minutes)", min_value=0.0, value=30000.0, step=100.0)
        video_length = st.number_input("Video Length (Minutes)", min_value=0.0, value=15.0, step=0.5)
        subscribers = st.number_input("Current Subscribers", min_value=0, value=100000, step=1000)
        year = st.selectbox("Year of Post", [2024, 2025, 2026])

    with col2:
        st.subheader("🌍 Metadata")
        category = st.selectbox("Content Category", CATEGORIES)
        device = st.selectbox("Target Device", DEVICES)
        country = st.selectbox("Target Country", COUNTRIES)
        
        # Calculate Engagement Rate (Engineered Feature)
        # Formula: (Likes + Comments) / Views
        engagement_rate = round((likes + comments) / views, 4) if views > 0 else 0.0
        st.metric("Engagement Rate", f"{engagement_rate*100:.2f}%")
        
        st.write("---")
        
        if st.button("Calculate Revenue"):
            if model is not None:
                # Prepare input data dictionary with all necessary columns initialized to 0
                input_data = {
                    'views': views,
                    'likes': likes,
                    'comments': comments,
                    'watch_time_minutes': watch_time,
                    'video_length_minutes': video_length,
                    'subscribers': subscribers,
                    'Year': year,
                    'engagement_rate': engagement_rate,
                }
                
                # One-Hot Encoding Logic:
                # We need to match the exact features the model was trained on.
                # Since we used pd.get_dummies or OneHotEncoder during training, 
                # we must manually set the active category/device/country to 1 and others to 0.
                
                # Add one-hot encoded columns (initialized to 0)
                # Categories (Reference: Education)
                for cat in ["Entertainment", "Gaming", "Lifestyle", "Music", "Tech"]:
                    input_data[f'category_{cat}'] = 1 if category == cat else 0
                    
                # Devices (Reference: Desktop)
                for dev in ["Mobile", "TV", "Tablet"]:
                    input_data[f'device_{dev}'] = 1 if device == dev else 0
                    
                # Countries (Reference: AU)
                for ctry in ["CA", "DE", "IN", "UK", "US"]:
                    input_data[f'country_{ctry}'] = 1 if country == ctry else 0

                # Create DataFrame with exact column order
                feature_order = [
                    'views', 'likes', 'comments', 'watch_time_minutes', 'video_length_minutes', 'subscribers', 'Year',
                    'category_Entertainment', 'category_Gaming', 'category_Lifestyle', 'category_Music', 'category_Tech',
                    'device_Mobile', 'device_TV', 'device_Tablet',
                    'country_CA', 'country_DE', 'country_IN', 'country_UK', 'country_US',
                    'engagement_rate'
                ]
                
                input_df = pd.DataFrame([input_data])[feature_order]
                
                # Make Prediction
                # Returns an array, so we take the first element [0]
                prediction = model.predict(input_df)[0]
                
                st.balloons()
                st.success(f"### Estimated Ad Revenue: ${prediction:,.2f}")
                st.info("Note: This is an estimate based on historical data patterns.")
            else:
                st.error("Model not loaded. Please check if 'model.pkl' exists.")

elif page == "Data Insights":
    st.title("📊  Data Analysis")
    st.markdown("EDA (Exploratory Data Analysis): Identify trends, correlations, and outliers.")
    
    if not df_sample.empty:
        # Create Tabs for better organization
        tab1, tab2, tab3, tab4 = st.tabs(["🔥 Correlation", "📈 Distributions", "📅 Trends", "🏷️ Categories"])
        
        # --- Tab 1: Correlation ---
        with tab1:
            st.header("Correlation Heatmap")
            st.write("How do different metrics relate to each other?")
            
            # Select numerical columns
            numeric_df = df_sample.select_dtypes(include=[np.number])
            
            # Allow user to pick columns for correlation
            corr_cols = st.multiselect(
                "Select features for correlation:",
                options=numeric_df.columns.tolist(),
                default=['views', 'likes', 'comments', 'watch_time_minutes', 'ad_revenue_usd']
            )
            
            if len(corr_cols) > 1:
                fig_corr, ax_corr = plt.subplots(figsize=(10, 8))
                sns.heatmap(
                    df_sample[corr_cols].corr(), 
                    annot=True, 
                    cmap='coolwarm', 
                    fmt='.2f', 
                    ax=ax_corr,
                    linewidths=0.5
                )
                st.pyplot(fig_corr)
            else:
                st.info("Please select at least two columns to see the correlation.")

        # --- Tab 2: Distributions & Outliers ---
        with tab2:
            st.header("Distributions & Outliers")
            
            dist_col = st.selectbox(
                "Select a Metric to Visualize:",
                ['ad_revenue_usd', 'views', 'likes', 'comments', 'watch_time_minutes']
            )
            
            col_d1, col_d2 = st.columns(2)
            
            with col_d1:
                st.subheader("Histogram (Distribution)")
                fig_hist, ax_hist = plt.subplots()
                sns.histplot(df_sample[dist_col], kde=True, color='skyblue', ax=ax_hist)
                ax_hist.set_title(f"Distribution of {dist_col}")
                st.pyplot(fig_hist)
                
            with col_d2:
                st.subheader("Box Plot (Outliers)")
                fig_box, ax_box = plt.subplots()
                sns.boxplot(y=df_sample[dist_col], color='lightcoral', ax=ax_box)
                ax_box.set_title(f"Box Plot of {dist_col}")
                st.pyplot(fig_box)
                
            # Statistical Highlights for the selected column
            st.write(f"**Key Statistics for {dist_col}**")
            st.dataframe(df_sample[dist_col].describe().to_frame().T)

        # --- Tab 3: Trends Over Time ---
        with tab3:
            st.header("Temporal Trends")
            
            if 'Date' in df_sample.columns:
                # Group by Month or Day (Sample data might be sparse, so Day is safe)
                trend_metric = st.selectbox("Select Metric for Trend:", ['ad_revenue_usd', 'views'], index=0)
                
                # Resample or Group
                # Since we loaded partial data, we just sort by date
                df_sorted = df_sample.sort_values(by='Date')
                
                fig_trend, ax_trend = plt.subplots(figsize=(12, 5))
                sns.lineplot(data=df_sorted, x='Date', y=trend_metric, ax=ax_trend, color='green')
                ax_trend.set_title(f"{trend_metric} Over Time")
                plt.xticks(rotation=45)
                st.pyplot(fig_trend)
            else:
                st.warning("Date column not found or invalid.")

        # --- Tab 4: Categorical Analysis ---
        with tab4:
            st.header("Categorical Insights")
            
            cat_option = st.selectbox("Analyze by:", ["Category", "Device", "Country"])
            cat_col_map = {"Category": "category", "Device": "device", "Country": "country"}
            selected_cat_col = cat_col_map[cat_option]
            
            col_c1, col_c2 = st.columns(2)
            
            with col_c1:
                st.subheader(f"Revenue by {cat_option}")
                # Bar Chart of Average Revenue
                fig_cat, ax_cat = plt.subplots()
                avg_rev_cat = df_sample.groupby(selected_cat_col)['ad_revenue_usd'].mean().sort_values(ascending=False)
                sns.barplot(x=avg_rev_cat.values, y=avg_rev_cat.index, palette='viridis', ax=ax_cat)
                ax_cat.set_xlabel("Avg Ad Revenue (USD)")
                st.pyplot(fig_cat)
                
            with col_c2:
                st.subheader(f"Count of {cat_option}")
                fig_count, ax_count = plt.subplots()
                sns.countplot(y=df_sample[selected_cat_col], order=df_sample[selected_cat_col].value_counts().index, palette='pastel', ax=ax_count)
                st.pyplot(fig_count)

    else:
        st.warning("Dataset not found for insights.")

elif page == "Model Insights":
    st.title("🧠 Model Logic & Feature Importance")
    st.markdown("Discover which factors most heavily influence the model's revenue predictions.")
    
    if model is not None:
        try:
            # Extract Feature Importance
            importance = model.feature_importances_
            feature_names = model.feature_names_in_
            
            # Create DataFrame
            fi_df = pd.DataFrame({
                'Feature': feature_names,
                'Importance': importance
            }).sort_values(by='Importance', ascending=False)
            
            # Highlight top features
            st.success(f"🌟 **Top Influencer**: {fi_df.iloc[0]['Feature']} (Importance: {fi_df.iloc[0]['Importance']:.2%})")
            
            # Visualization
            st.subheader("Feature Importance Ranking")
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.barplot(data=fi_df, x='Importance', y='Feature', palette="viridis", ax=ax)
            ax.set_title("Which features matter most?")
            ax.set_xlabel("Relative Importance (0-1)")
            st.pyplot(fig)
            
            st.write("### Interpretation")
            st.info("""
            - **High Importance**: Changing these values will have a drastic effect on the predicted revenue.
            - **Low Importance**: Variable changes here might not significantly alter the outcome.
            """)
            
        except Exception as e:
            st.error(f"Could not extract feature importance: {e}")
    else:
        st.error("Model not loaded.")


# Footer
st.sidebar.write("---")
# Footer section with project credits
st.sidebar.caption("Created for Content Monetization Project")
# End of Application