import streamlit as st
import pandas as pd
import numpy as np
import xgboost as xgb
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import os

st.set_page_config(page_title="Sales Analytics Dashboard", layout="wide")

@st.cache_data
def load_data():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(base_dir, "train.csv")
    
    # encoding='utf-8-sig' handles hidden BOM characters automatically
    data = pd.read_csv(file_path, encoding='utf-8-sig')
    
    # Clean up column spaces and convert to lowercase for uniform matching
    data.columns = data.columns.str.strip().str.lower()
    
    # Helper to fuzzy find columns safely
    def find_col(possible_names):
        for col in data.columns:
            if col in possible_names:
                return col
        raise KeyError(f"Could not find any columns matching: {possible_names}")

    # Map core metrics dynamically
    sales_col = find_col(['sales', 'sales volume', 'total sales', 'amount'])
    region_col = find_col(['region', 'zone', 'location'])
    category_col = find_col(['category', 'cat'])
    subcat_col = find_col(['sub-category', 'subcategory', 'sub_category'])
    
    # Rename basic columns to standard layout
    data = data.rename(columns={
        sales_col: 'Sales',
        region_col: 'Region',
        category_col: 'Category',
        subcat_col: 'Sub-Category'
    })

    # Date parsing block with dynamic fallbacks
    try:
        # Check if individual split columns exist
        year_col = find_col(['order year', 'year', 'order_year'])
        month_col = find_col(['order month', 'month', 'order_month'])
        week_col = find_col(['order week', 'week', 'order_week'])
        
        # Strip floats down by filling missing values and casting cleanly to int
        data['Order Year'] = pd.to_numeric(data[year_col], errors='coerce').fillna(0).astype(int)
        data['Order Month'] = pd.to_numeric(data[month_col], errors='coerce').fillna(1).astype(int)
        data['Order Week'] = pd.to_numeric(data[week_col], errors='coerce').fillna(1).astype(int)
        
    except KeyError:
        # Fallback: Parse from a singular date string column if explicit elements are missing
        date_col = find_col(['order date', 'date', 'order_date', 'timestamp'])
        parsed_dates = pd.to_datetime(data[date_col], errors='coerce')
        
        data['Order Year'] = parsed_dates.dt.year.fillna(0).astype(int)
        data['Order Month'] = parsed_dates.dt.month.fillna(1).astype(int)
        data['Order Week'] = parsed_dates.dt.isocalendar().week.fillna(1).astype(int)
        
    # Build timeline coordinates safely using the newly sanitized integers
    data['ds'] = pd.to_datetime(data['Order Year'].astype(str) + '-' + data['Order Month'].astype(str) + '-01', errors='coerce')
    data['ds_week'] = pd.to_datetime(data['Order Year'].astype(str) + '-' + data['Order Week'].astype(str) + '-1', format='%G-%V-%u', errors='coerce')
    
    # Drop rows where dates failed to parse to prevent chart breakdowns
    data = data.dropna(subset=['ds'])
    return data

# Run data loader
data = load_data()

st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Page 1 — Historical Performance", "Page 2 — Forecast Explorer", "Page 3 — Anomaly Report", "Page 4 — Product Demand Segments"])

# ==========================================
# PAGE 1 CODE
# ==========================================
if page == "Page 1 — Historical Performance":
    st.title("Historical Sales Performance")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Total Sales by Year")
        yearly_sales = data.groupby('Order Year')['Sales'].sum().reset_index()
        fig, ax = plt.subplots()
        ax.bar(yearly_sales['Order Year'].astype(str), yearly_sales['Sales'], color='skyblue')
        ax.set_xlabel("Year")
        ax.set_ylabel("Total Sales")
        st.pyplot(fig)
        
    with col2:
        st.subheader("Monthly Sales Trend")
        monthly_sales = data.groupby('ds')['Sales'].sum().reset_index()
        fig, ax = plt.subplots()
        ax.plot(monthly_sales['ds'], monthly_sales['Sales'], marker='o', color='orange')
        ax.set_xlabel("Date")
        ax.set_ylabel("Sales")
        plt.xticks(rotation=45)
        st.pyplot(fig)
        
    st.markdown("---")
    st.subheader("Interactive Sales Explorer")
    selected_region = st.selectbox("Select Region", ["All"] + list(data['Region'].unique()))
    selected_category = st.selectbox("Select Category", ["All"] + list(data['Category'].unique()))
    
    filtered_data = data.copy()
    if selected_region != "All":
        filtered_data = filtered_data[filtered_data['Region'] == selected_region]
    if selected_category != "All":
        filtered_data = filtered_data[filtered_data['Category'] == selected_category]
        
    st.dataframe(filtered_data.groupby(['Region', 'Category'])['Sales'].sum().reset_index())

# ==========================================
# PAGE 2 CODE
# ==========================================
elif page == "Page 2 — Forecast Explorer":
    st.title("Operational Forecast Explorer")
    
    dimension = st.selectbox("Forecast Dimension", ["Category", "Region"])
    
    if dimension == "Category":
        options = data['Category'].unique()
    else:
        options = data['Region'].unique()
        
    selected_value = st.selectbox(f"Select {dimension}", options)
    horizon = st.slider("Select Forecast Horizon (Months)", 1, 3, 3)
    
    if dimension == "Category":
        seg_data = data[data['Category'] == selected_value]
    else:
        seg_data = data[data['Region'] == selected_value]
        
    df_ml = seg_data.groupby('ds')['Sales'].sum().reset_index()
    df_ml.set_index('ds', inplace=True)
    
    if len(df_ml) >= 6:
        df_ml['lag_1'] = df_ml['Sales'].shift(1)
        df_ml['lag_2'] = df_ml['Sales'].shift(2)
        df_ml['lag_3'] = df_ml['Sales'].shift(3)
        df_ml['rolling_mean_3'] = df_ml['Sales'].shift(1).rolling(window=3).mean()
        df_ml['month'] = df_ml.index.month
        df_ml['quarter'] = df_ml.index.quarter
        df_ml = df_ml.dropna()
        
        features = ['lag_1', 'lag_2', 'lag_3', 'rolling_mean_3', 'month', 'quarter']
        X = df_ml[features]
        y = df_ml['Sales']
        
        X_train, X_test = X.iloc[:-3], X.iloc[-3:]
        y_train, y_test = y.iloc[:-3], y.iloc[-3:]
        
        model = xgb.XGBRegressor(n_estimators=100, learning_rate=0.05, max_depth=5, random_state=42)
        model.fit(X_train, y_train)
        
        predictions = model.predict(X_test)[:horizon]
        pred_index = y_test.index[:horizon]
        
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(df_ml.index[-12:], df_ml['Sales'].tail(12), label='Actual Sales', marker='o')
        ax.plot(pred_index, predictions, label='XGBoost Forecast', marker='x', linestyle='--', color='red')
        ax.set_title(f"XGBoost Forecast for {selected_value}")
        ax.legend()
        st.pyplot(fig)
        
        st.subheader("Forecast Values")
        st.write(pd.DataFrame({'Date': pred_index, 'Forecasted Sales': predictions}).set_index('Date'))
    else:
        st.warning("Not enough historical data points to generate an XGBoost forecast for this segment.")

# ==========================================
# PAGE 3 CODE
# ==========================================
elif page == "Page 3 — Anomaly Report":
    st.title("Sales Anomaly Detection Report")
    
    st.subheader("Monthly Level Anomalies (Isolation Forest)")
    df_monthly = data.groupby('ds')['Sales'].sum().reset_index()
    df_monthly.set_index('ds', inplace=True)
    
    iso_forest = IsolationForest(contamination=0.05, random_state=42)
    df_monthly['anomaly_score'] = iso_forest.fit_predict(df_monthly[['Sales']])
    df_monthly['is_anomaly'] = df_monthly['anomaly_score'] == -1
    
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(df_monthly.index, df_monthly['Sales'], color='blue', label='Monthly Sales')
    anomalies_m = df_monthly[df_monthly['is_anomaly']]
    ax.scatter(anomalies_m.index, anomalies_m['Sales'], color='red', label='Anomaly', zorder=5)
    ax.legend()
    st.pyplot(fig)
    
    st.write(df_monthly[df_monthly['is_anomaly']][['Sales']])
    
    st.markdown("---")
    st.subheader("Weekly Level Anomalies (Rolling Z-Score)")
    
    # Filter out any dynamic weekly conversion failures
    df_weekly_clean = data.dropna(subset=['ds_week'])
    df_weekly = df_weekly_clean.groupby('ds_week')['Sales'].sum().reset_index()
    df_weekly.set_index('ds_week', inplace=True)
    
    if len(df_weekly) > 4:
        df_weekly['rolling_mean'] = df_weekly['Sales'].rolling(window=4).mean()
        df_weekly['rolling_std'] = df_weekly['Sales'].rolling(window=4).std()
        df_weekly['z_score'] = (df_weekly['Sales'] - df_weekly['rolling_mean']) / df_weekly['rolling_std']
        df_weekly['is_anomaly'] = df_weekly['z_score'].abs() > 2.0
        
        fig, ax = plt.subplots(figsize=(12, 4))
        ax.plot(df_weekly.index, df_weekly['Sales'], color='green', label='Weekly Sales')
        anomalies_w = df_weekly[df_weekly['is_anomaly']]
        ax.scatter(anomalies_w.index, anomalies_w['Sales'], color='red', label='Anomaly', zorder=5)
        ax.legend()
        st.pyplot(fig)
        
        st.write(df_weekly[df_weekly['is_anomaly']][['Sales', 'z_score']])
    else:
        st.info("Insufficient weekly intervals available to plot rolling anomalies.")

# ==========================================
# PAGE 4 CODE
# ==========================================
elif page == "Page 4 — Product Demand Segments":
    st.title("Product Sub-Category Demand Segments")
    
    product_features = data.groupby('Sub-Category').agg(
        total_sales_volume=('Sales', 'sum'),
        average_order_value=('Sales', 'mean'),
        sales_volatility=('Sales', 'std')
    ).reset_index()
    product_features['sales_volatility'] = product_features['sales_volatility'].fillna(0)
    
    latest_year = data['Order Year'].max()
    prior_year = latest_year - 1
    sales_latest = data[data['Order Year'] == latest_year].groupby('Sub-Category')['Sales'].sum()
    sales_prior = data[data['Order Year'] == prior_year].groupby('Sub-Category')['Sales'].sum()
    growth_rate = ((sales_latest - sales_prior) / sales_prior).fillna(0).replace([np.inf, -np.inf], 0)
    growth_rate = growth_rate.reset_index().rename(columns={'Sales': 'sales_growth_rate'})
    product_features = pd.merge(product_features, growth_rate, on='Sub-Category', how='left').fillna(0)
    
    features = ['total_sales_volume', 'sales_growth_rate', 'sales_volatility', 'average_order_value']
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(product_features[features])
    
    optimal_k = 4
    kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
    product_features['Cluster'] = kmeans.fit_predict(scaled_features)
    
    cluster_map = {
        0: 'High Volume, Stable Demand',
        1: 'Low Volume, High Volatility',
        2: 'Growing Demand',
        3: 'Declining Demand'
    }
    product_features['Demand_Segment'] = product_features['Cluster'].map(cluster_map)
    
    fig, ax = plt.subplots(figsize=(10, 5))
    for cluster_num in range(optimal_k):
        clustered_data = product_features[product_features['Cluster'] == cluster_num]
        ax.scatter(clustered_data['total_sales_volume'], clustered_data['sales_volatility'], label=cluster_map[cluster_num], alpha=0.7)
    ax.set_xlabel('Total Sales Volume')
    ax.set_ylabel('Sales Volatility (Std Dev)')
    ax.legend()
    st.pyplot(fig)
    
    st.subheader("Sub-Category Assignments")
    st.dataframe(product_features[['Sub-Category', 'Demand_Segment']].sort_values(by='Demand_Segment'))