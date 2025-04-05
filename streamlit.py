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
data_path = "Online Retail.csv"

if os.path.exists(data_path):
    df = pd.read_csv(data_path)
    st.success("✅ Dataset loaded successfully!")
    st.write(df.head())
else:
    st.warning(f"Please upload or add `{data_path}` to your project directory.")

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
