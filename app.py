import streamlit as st
import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import statsmodels.api as sm
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import plotly.express as px

# --- USER AUTHENTICATION SETUP ---
USER_CREDENTIALS = {"admin": "password123", "user": "pass456", "123": "123"}

# Initialize session state variables
if "authenticated" not in st.session_state:
    st.session_state["authenticated"] = False

# Login function
def login():
    st.set_page_config(page_title='Data Visualizer', layout='centered', page_icon='📊')
    st.title("🔐 Login")
    username = st.text_input("Username")
    password = st.text_input("Password", type="password")
    if st.button("Login"):
        if username in USER_CREDENTIALS and USER_CREDENTIALS[username] == password:
            st.session_state["authenticated"] = True
            st.session_state["user"] = username
            st.success("Login successful!")
            st.rerun()
        else:
            st.error("Invalid username or password.")

# Logout function
def logout():
    st.session_state["authenticated"] = False
    st.session_state["user"] = None
    st.rerun()

# --- LOGIN PAGE ---
if not st.session_state["authenticated"]:
    login()
    st.stop()

# --- MAIN APP AFTER LOGIN ---
st.set_page_config(page_title='Data Visualizer', layout='wide', page_icon='📊')

st.sidebar.title(f"👤 Welcome, {st.session_state['user']}!")
if st.sidebar.button("Logout"):
    logout()

st.title('Interactive Demographic Statistics Visualizer')

# --- FILE HANDLING ---
folder_path = "data"
if not os.path.exists(folder_path):
    os.makedirs(folder_path)

uploaded_file = st.file_uploader("📂 Upload a CSV file", type=["csv"])

if uploaded_file:
    file_path = os.path.join(folder_path, uploaded_file.name)
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    df = pd.read_csv(file_path)
    st.success("File uploaded successfully!")
else:
    files = [f for f in os.listdir(folder_path) if f.endswith('.csv')]
    if not files:
        st.warning("⚠ No CSV files found in the data folder. Please upload a file.")
        st.stop()
    selected_file = st.selectbox('Select a file', files)
    df = pd.read_csv(os.path.join(folder_path, selected_file))

# --- DATA PREVIEW ---
st.write("### Data Preview")
st.write(df.head())

columns = df.columns.tolist()
x_axis = st.selectbox(' Select the X-axis', options=["None"] + columns)
y_axis = st.selectbox(' Select the Y-axis', options=["None"] + columns)

# --- PLOT SELECTION ---
plot_list = ['Line Plot', 'Bar Chart', 'Scatter Plot', 'Distribution Plot', 'Count Plot', 'Box Plot', 'Violin Plot', 'Correlation Heatmap', 'Pairplot']
plot_type = st.selectbox(' Select the type of plot', options=plot_list)

if st.button('Generate Plot'):
    if x_axis == "None":
        st.error("⚠ Please select a valid X-axis variable.")
    elif plot_type in ["Line Plot", "Bar Chart", "Scatter Plot"] and y_axis == "None":
        st.error("⚠ Please select a valid Y-axis variable for this plot.")
    else:
        fig, ax = plt.subplots(figsize=(6, 4))
        if plot_type == 'Line Plot':
            sns.lineplot(x=df[x_axis], y=df[y_axis], ax=ax)
        elif plot_type == 'Bar Chart':
            sns.barplot(x=df[x_axis], y=df[y_axis], ax=ax)
        elif plot_type == 'Scatter Plot':
            sns.scatterplot(x=df[x_axis], y=df[y_axis], ax=ax)
        elif plot_type == 'Distribution Plot':
            sns.histplot(df[x_axis], kde=True, ax=ax)
            ax.set_ylabel("Density")
        elif plot_type == 'Count Plot':
            sns.countplot(x=df[x_axis], ax=ax)
            ax.set_ylabel("Count")
        elif plot_type == 'Box Plot':
            sns.boxplot(x=df[x_axis], y=df[y_axis], ax=ax)
        elif plot_type == 'Violin Plot':
            sns.violinplot(x=df[x_axis], y=df[y_axis], ax=ax)
        elif plot_type == 'Correlation Heatmap':
            fig, ax = plt.subplots(figsize=(8, 6))
            sns.heatmap(df.corr(), annot=True, cmap='coolwarm', ax=ax)
        elif plot_type == 'Pairplot':
            fig = sns.pairplot(df)

        ax.set_title(f'{plot_type} of {y_axis} vs {x_axis}', fontsize=12)
        st.pyplot(fig)

# --- ADVANCED VISUALIZATIONS ---
st.write("## ADVANCED VISUALIZATION ")
if df[columns[2]].dtype == 'object':
    df[columns[2]] = df[columns[2]].astype('category').cat.codes
fig = px.sunburst(df, path=[columns[0], columns[1]], values=columns[2])
st.plotly_chart(fig)

# --- PREDICTIVE ANALYSIS ---
st.write("## Predictive Analysis")
predict_x = st.selectbox("📈 Select feature (X) for prediction", options=columns)
predict_y = st.selectbox("📉 Select target (Y) for prediction", options=columns)

if st.button("Run Prediction"):
    X = df[[predict_x]].copy()
    y = df[predict_y].copy()

    # Handle categorical data
    if X[predict_x].dtype == 'object':
        le = LabelEncoder()
        X[predict_x] = le.fit_transform(X[predict_x])

    if y.dtype == 'object':
        le_y = LabelEncoder()
        y = le_y.fit_transform(y)

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train model
    model = LinearRegression()
    model.fit(X_train, y_train)

    # Predict
    predictions = model.predict(X_test)

    st.write(f"### Prediction for {predict_y} based on {predict_x}")
    results_df = pd.DataFrame({"Actual": y_test, "Predicted": predictions})
    st.write(results_df.head())

    # Plot predictions
    fig, ax = plt.subplots()
    sns.regplot(x=y_test, y=predictions, ax=ax)
    ax.set_title("Actual vs Predicted Values")
    st.pyplot(fig)

# --- TIME-SERIES FORECASTING ---
st.write("## Time-Series Forecasting")
time_col = st.selectbox("🗓️ Select time column for forecasting", options=columns)
target_col = st.selectbox("📊 Select target column for forecasting", options=columns)

if st.button("Run Forecasting"):
    try:
        # Convert time column to datetime and sort
        df[time_col] = pd.to_datetime(df[time_col], errors='coerce')
        df = df.sort_values(by=time_col)
        df = df.set_index(time_col)

        # Convert target column to numeric
        df[target_col] = pd.to_numeric(df[target_col], errors='coerce')

        # Handle missing values
        df[target_col].fillna(method='ffill', inplace=True)
        df.dropna(subset=[target_col], inplace=True)

        # Build ARIMA model
        model = sm.tsa.ARIMA(df[target_col], order=(5,1,0))
        results = model.fit()
        forecast = results.forecast(steps=10)

        # Display results
        st.write("### Forecasted Values")
        st.write(forecast)

        # Plot forecast
        fig, ax = plt.subplots()
        ax.plot(df.index, df[target_col], label="Actual")
        ax.plot(pd.date_range(df.index[-1], periods=10, freq='M'), forecast, label="Forecast", linestyle='dashed')
        ax.legend()
        st.pyplot(fig)

    except Exception as e:
        st.error(f"❌ Forecasting failed: {e}")
