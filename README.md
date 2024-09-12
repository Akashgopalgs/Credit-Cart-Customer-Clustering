# 💳 Credit Card Customer Clustering 
![Screenshot 2024-09-12 185852](https://github.com/user-attachments/assets/46f5e01c-1808-4fa4-ac56-03c63fb5dbf9)


This project uses unsupervised machine learning algorithms to analyze and cluster credit card customer data. The main objective is to segment customers based on their credit card usage patterns using clustering techniques like DBSCAN and PCA. 

## 🚀 Overview
This project focuses on customer segmentation using credit card transaction data, clustering customers based on their spending patterns and behaviors. It includes the following steps:

- Data cleaning and preprocessing
- Exploratory Data Analysis (EDA)
- Clustering using DBSCAN and dimensionality reduction using PCA
- Building a web-based interface using Streamlit for interactive customer input 

## 🧠 Model Explanation
### PCA (Principal Component Analysis):
- PCA was used to reduce the dimensionality of the dataset for better visualization and interpretation of the clusters.
- Transformed the data into two principal components for plotting and easier visualization.


### DBSCAN (Density-Based Spatial Clustering of Applications with Noise):
- DBSCAN is used to discover clusters in the dataset by identifying high-density regions.
- Parameters such as eps (radius of the neighborhood) and min_samples (minimum number of points required to form a cluster) were tuned to achieve optimal results.
- The clustering results were evaluated using metrics such as Silhouette Score and Calinski-Harabasz Score.

## 💻 Streamlit Application 
The application allows users to:

- Input custom data for a new credit card customer.
- Predict the cluster that the new customer would belong to, based on their spending habits.
- APP : https://credit-cart-customer-clustering-ft7nawsdxgxau5g74xtbnb.streamlit.app/
