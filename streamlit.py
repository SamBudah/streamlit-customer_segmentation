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
import pickle
import os

# Set page configuration for a wider layout and custom theme
st.set_page_config(page_title="Customer Segmentation Dashboard", layout="wide", initial_sidebar_state="expanded")

# Custom CSS for styling
st.markdown("""
    <style>
    .main {
        background-color: #f0f2f6;
        padding: 20px;
    }
    .stButton>button {
        background-color: #4CAF50;
        color: white;
        border-radius: 5px;
    }
    .stButton>button:hover {
        background-color: #45a049;
    }
    .stSidebar {
        background-color: #f0f2f6;
    }
    </style>
""", unsafe_allow_html=True)

# Sidebar for navigation and settings
st.sidebar.title("🧠 Customer Segmentation Dashboard")
st.sidebar.markdown("Built with KMeans and Random Forest")
page = st.sidebar.selectbox("Navigate", ["Home", "Data Upload & Preprocessing", "Clustering", "Classification", "Insights & Recommendations"])

# Main content
if page == "Home":
    st.title("🧠 Customer Segmentation Dashboard")
    st.markdown("""
    Welcome to the Customer Segmentation Dashboard! This app helps Kenyan cooperatives (Coop Affairs) segment customers for targeted marketing. 
    - **Upload** your customer data.
    - **Preprocess** and clean the data.
    - **Cluster** customers using KMeans.
    - **Classify** new customers with a Random Forest model.
    - **Gain Insights** and recommendations for marketing strategies.
    """)

elif page == "Data Upload & Preprocessing":
    st.header("📂 Data Upload & Preprocessing")
    st.subheader("Upload your customer data file")
    uploaded_file = st.file_uploader("Drag and drop file here", type=['csv', 'xlsx'], help="Limit 200MB per file • CSV, XLSX")

    if uploaded_file is not None:
        # Load the dataset
        df = pd.read_csv(uploaded_file)
        st.success("Dataset loaded successfully!")

        # Display the dataset
        st.subheader("First 5 rows of the dataset:")
        st.dataframe(df.head())

        # Data Preprocessing
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

        # Save the processed dataset to session state for use in other pages
        st.session_state['df'] = df
        st.session_state['features'] = features
        st.session_state['scaler'] = scaler

elif page == "Clustering":
    st.header("🔄 KMeans Clustering")
    if 'df' not in st.session_state:
        st.warning("Please upload and preprocess the dataset first on the 'Data Upload & Preprocessing' page.")
    else:
        df = st.session_state['df']
        features = st.session_state['features']

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

        # Save the updated dataset with clusters
        st.session_state['df'] = df

        # Download button for the processed dataset
        csv = df.to_csv(index=False)
        st.download_button(
            label="Download Processed Dataset with Clusters",
            data=csv,
            file_name="processed_customer_data.csv",
            mime="text/csv"
        )

elif page == "Classification":
    st.header("🤖 Random Forest Classification")
    if 'df' not in st.session_state:
        st.warning("Please complete the clustering step on the 'Clustering' page first.")
    else:
        df = st.session_state['df']
        scaler = st.session_state['scaler']

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

        # Save the trained model and scaler for deployment
        with open('rf_model.pkl', 'wb') as file:
            pickle.dump(rf_model, file)
        with open('scaler.pkl', 'wb') as file:
            pickle.dump(scaler, file)
        st.success("Trained model and scaler saved successfully for deployment!")

elif page == "Insights & Recommendations":
    st.header("💡 Insights & Recommendations")
    if 'df' not in st.session_state:
        st.warning("Please complete the clustering and classification steps first.")
    else:
        # Insights
        with st.expander("Insights", expanded=True):
            st.write("""
            - **Cluster 0**: Moderate precision and recall, indicating potential overlap with other segments. Marketing strategies should focus on distinguishing this segment more clearly.
            - **Cluster 1**: High recall and precision, making it the most consistently identified group. This segment is highly predictable, possibly indicating loyal or consistent purchasing behavior.
            - **Cluster 2**: Although the smallest group, it maintained high precision and recall, suggesting a niche but well-defined segment.
            - **Cluster 3**: High precision, making it a distinct segment from others, ideal for targeted campaigns.
            """)

        # Recommendations
        with st.expander("Recommendations"):
            st.write("""
            - **Marketing Strategy**: Tailor marketing campaigns to target each segment more effectively, maximizing customer engagement and conversion rates.
            - **Business Decision-Making**: Utilize these segments for product recommendations, personalized offers, and strategic inventory management.
            """)
