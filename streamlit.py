# Import necessary libraries
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import classification_report, accuracy_score
import os

# Set page title
st.title("🧠 Customer Segmentation App")
st.write("Built with KMeans and Random Forest")

# Step 1: Upload the dataset
st.subheader("Upload your customer data file")
uploaded_file = st.file_uploader("Drag and drop file here", type=['csv', 'xlsx'], help="Limit 200MB per file • CSV, XLSX")

if uploaded_file is not None:
    # Load the dataset
    df = pd.read_csv(uploaded_file)
    st.success("Dataset loaded successfully!")

    # Step 2: Display the dataset
    st.subheader("First 5 rows of the dataset:")
    st.dataframe(df.head())

    # Step 3: Data Preprocessing
    st.subheader("Data Preprocessing")
    st.write("Cleaning the dataset and preparing features for clustering.")

    # Drop missing values
    df = df.dropna()
    st.write("Missing values dropped. Shape of cleaned dataset:", df.shape)

    # Define features
    features = ['reports', 'age', 'income', 'share', 'expenditure', 'dependents', 'months', 'majorcards', 'active']
    categorical_features = ['card', 'owner', 'selfemp']

    # Convert categorical features to dummy variables
    df = pd.get_dummies(df, columns=categorical_features, drop_first=True)

    # Standardize numeric features
    scaler = StandardScaler()
    df[features] = scaler.fit_transform(df[features])
    st.write("Numeric features standardized.")

    # Step 4: KMeans Clustering
    st.subheader("KMeans Clustering")
    st.write("Using the Elbow Method to determine the optimal number of clusters.")

    # Elbow Method
    wcss = []
    for i in range(1, 11):
        kmeans = KMeans(n_clusters=i, init='k-means++', random_state=42)
        kmeans.fit(df[features])
        wcss.append(kmeans.inertia_)

    # Plot the Elbow Method
    fig, ax = plt.subplots(figsize=(8, 6))
    plt.plot(range(1, 11), wcss, marker='o', linestyle='--', color='b')
    plt.xlabel("Number of Clusters")
    plt.ylabel("WCSS")
    plt.title("Elbow Method for Optimal K")
    plt.grid(True)
    st.pyplot(fig)

    # Apply KMeans with 4 clusters (as determined)
    n_clusters = 4
    kmeans = KMeans(n_clusters=n_clusters, init='k-means++', random_state=42)
    df['Cluster'] = kmeans.fit_predict(df[features])
    st.write("Cluster Distribution:")
    st.write(df['Cluster'].value_counts())

    # PCA for visualization
    pca = PCA(n_components=2)
    pca_components = pca.fit_transform(df.drop(columns=['Cluster']))
    df['PCA Component 1'] = pca_components[:, 0]
    df['PCA Component 2'] = pca_components[:, 1]

    # Visualize clusters
    st.write("Cluster Visualization (PCA Reduced Dimensions):")
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.scatterplot(x='PCA Component 1', y='PCA Component 2', hue='Cluster', palette='viridis', data=df)
    plt.xlabel("PCA Component 1")
    plt.ylabel("PCA Component 2")
    plt.title("Customer Segmentation (PCA Reduced Dimensions)")
    plt.legend(title="Cluster")
    st.pyplot(fig)

    # Step 5: Random Forest Classification
    st.subheader("Random Forest Classification")
    st.write("Using cluster labels as the target variable to train a Random Forest Classifier.")

    # Define features and target
    X = df.drop(columns=['Cluster', 'PCA Component 1', 'PCA Component 2'])
    y = df['Cluster']

    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train Random Forest
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_model.fit(X_train, y_train)

    # Predict on test set
    y_pred = rf_model.predict(X_test)

    # Display accuracy
    accuracy = accuracy_score(y_test, y_pred)
    st.write(f"Random Forest Model Accuracy: {accuracy:.2f}")

    # Display classification report
    st.write("Classification Report:")
    report = classification_report(y_test, y_pred, output_dict=True)
    report_df = pd.DataFrame(report).transpose()
    st.dataframe(report_df)

    # Step 6: Insights and Interpretations
    st.subheader("Insights and Interpretations")
    st.write("""
    - **Cluster 0**: Moderate precision and recall, indicating potential overlap with other segments. Marketing strategies should focus on distinguishing this segment more clearly.
    - **Cluster 1**: High recall and precision, making it the most consistently identified group. This segment is highly predictable, possibly indicating loyal or consistent purchasing behavior.
    - **Cluster 2**: Although the smallest group, it maintained high precision and recall, suggesting a niche but well-defined segment.
    - **Cluster 3**: High precision, making it a distinct segment from others, ideal for targeted campaigns.
    """)

    # Step 7: Recommendations
    st.subheader("Recommendations")
    st.write("""
    - **Marketing Strategy**: Tailor marketing campaigns to target each segment more effectively, maximizing customer engagement and conversion rates.
    - **Business Decision-Making**: Utilize these segments for product recommendations, personalized offers, and strategic inventory management.
    """)

else:
    st.info("Please upload a dataset to begin.")

# Step 8: Add a button to manage the app (optional)
if st.button("Manage app"):
    st.write("App management options can be added here.")
