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

# Step 1: File Upload
uploaded_file = st.file_uploader("Upload your customer data file", type=["csv", "xlsx"])

if uploaded_file is not None:
    # Read the file based on its extension
    if uploaded_file.name.endswith('.csv'):
        df = pd.read_csv(uploaded_file)
    else:
        df = pd.read_excel(uploaded_file)
    
    st.success("✅ Dataset loaded successfully!")
    st.write("First 5 rows of the dataset:")
    st.write(df.head())

    # Step 2: KMeans Clustering
    st.subheader("🔍 KMeans Clustering")
    
    # Check if required columns exist
    required_columns = ['Age', 'Annual Income (k$)', 'Spending Score (1-100)']
    if all(col in df.columns for col in required_columns):
        features = df[required_columns]
        kmeans = KMeans(n_clusters=5, random_state=0)
        df['Cluster'] = kmeans.fit_predict(features)

        st.write("Clustered Data Sample:")
        st.write(df[['CustomerID', 'Cluster']].head())

        st.write("Visualizing Clusters:")
        fig, ax = plt.subplots()
        sns.scatterplot(data=df, x='Annual Income (k$)', y='Spending Score (1-100)', 
                        hue='Cluster', palette='Set2', ax=ax)
        st.pyplot(fig)

        # Step 3: Random Forest
        st.subheader("🌲 Random Forest Classification")
        
        if 'Gender' in df.columns:
            df['Gender'] = df['Gender'].map({'Male': 0, 'Female': 1})
            rf = RandomForestClassifier()
            X = df[['Age', 'Annual Income (k$)', 'Gender']]
            y = df['Cluster']
            rf.fit(X, y)
            y_pred = rf.predict(X)
            acc = accuracy_score(y, y_pred)
            st.success(f"Random Forest Self-Validation Accuracy: {acc:.2f}")
        else:
            st.warning("Gender column not found - skipping Random Forest classification")
    else:
        st.error(f"Dataset is missing required columns: {required_columns}")
else:
    st.warning("⚠️ Please upload a CSV or Excel file to proceed.")
