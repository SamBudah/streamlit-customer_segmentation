import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import os
import zipfile
import json

# Title
st.title("🧠 Customer Segmentation App")
st.markdown("Built with KMeans and Random Forest")

# Step 1: Load Kaggle Dataset
st.subheader("📥 Downloading Dataset from Kaggle")
if "kaggle.json" in os.listdir():
    os.environ['KAGGLE_CONFIG_DIR'] = os.getcwd()
    if not os.path.exists("data"):
        os.mkdir("data")
    if not os.path.exists("data/Mall_Customers.csv"):
        os.system("kaggle datasets download -d vjchoudhary7/customer-segmentation-tutorial -p data")
        with zipfile.ZipFile("data/customer-segmentation-tutorial.zip", 'r') as zip_ref:
            zip_ref.extractall("data")
        st.success("✅ Dataset downloaded and extracted!")
    df = pd.read_csv("data/Mall_Customers.csv")
    st.write(df.head())
else:
    st.warning("Please upload your `kaggle.json` file to start.")

# Step 2: KMeans Clustering
if "df" in locals():
    st.subheader("🔍 KMeans Clustering")
    features = df[['Age', 'Annual Income (k$)', 'Spending Score (1-100)']]
    kmeans = KMeans(n_clusters=5, random_state=0)
    df['Cluster'] = kmeans.fit_predict(features)

    st.write("Clustered Data Sample:")
    st.write(df[['CustomerID', 'Cluster']].head())

    st.write("Visualizing Clusters:")
    fig, ax = plt.subplots()
    sns.scatterplot(data=df, x='Annual Income (k$)', y='Spending Score (1-100)', hue='Cluster', palette='Set2', ax=ax)
    st.pyplot(fig)

    # Step 3: Random Forest
    st.subheader("🌲 Random Forest Classification")
    df['Gender'] = df['Gender'].map({'Male': 0, 'Female': 1})
    rf = RandomForestClassifier()
    X = df[['Age', 'Annual Income (k$)', 'Gender']]
    y = df['Cluster']
    rf.fit(X, y)
    y_pred = rf.predict(X)
    acc = accuracy_score(y, y_pred)
    st.success(f"Random Forest Self-Validation Accuracy: {acc:.2f}")
