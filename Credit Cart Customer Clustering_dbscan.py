#!/usr/bin/env python
# coding: utf-8

# ### Import Libraries & Dataset

# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import plotly.express as px
import plotly.io as pio


import warnings
warnings.filterwarnings('ignore')

# Import requirement libraries and tools for modeling
from sklearn.preprocessing import StandardScaler
from sklearn.impute import KNNImputer
from sklearn.cluster import KMeans, MiniBatchKMeans, MeanShift, estimate_bandwidth, AffinityPropagation, Birch, DBSCAN, OPTICS, AgglomerativeClustering 
from yellowbrick.cluster import KElbowVisualizer
import scipy.cluster.hierarchy as sch
from sklearn.mixture import GaussianMixture, BayesianGaussianMixture
from sklearn.model_selection import GridSearchCV
from sklearn import set_config
from sklearn.metrics import silhouette_score, calinski_harabasz_score
from sklearn.decomposition import PCA
from itertools import product
from sklearn.neighbors import NearestNeighbors
from sklearn.manifold import TSNE


# In[2]:


df = pd.read_csv(r"C:\Users\akash\OneDrive\Documents\dataset\kaggle dataset\CreditCard_customer_dataset.zip")
df


# ### Overview of Dataset

# In[3]:


df.info()


# In[4]:


desc_cust = df.select_dtypes(include=['int64','float64']).describe().T
desc_cust_df = pd.DataFrame(index=desc_cust.index, columns=desc_cust.columns, data=desc_cust )

# f,ax = plt.subplots(figsize=(12,12),dpi=500)
#
# sns.heatmap(desc_cust_df, annot=True,cmap = "Reds", fmt= '.0f',
#             ax=ax,linewidths = 5, cbar = False,
#             annot_kws={"size": 16})
#
# ax.xaxis.set_ticks_position('top')
# plt.tick_params(left=False, top=False)
# plt.xticks(size = 18)
# plt.yticks(size = 18)
# plt.title("Descriptive Statistics", pad=30, x=0.31, y=1.02)
# plt.show()


# ### Cleaning Dataset

# In[5]:


# Drop the cust_id columns because we do not neet it
df = df.drop('CUST_ID', axis=1)


# In[6]:


def highlight_cells(val):
    """Highlight cells that is not zero"""
    f_color = '#D1382F' if val != 0 else ''  # Red
    bg_color = '#D59A99' if val != 0 else ''   # Pink
    w_font = 'bold'
    return 'background-color: {}; color: {}; font-weight: {}'.format(bg_color, f_color, w_font)

# Check missing values
is_nan = df.isna().sum().to_frame(name='Number_of_NaN')
is_nan.insert(1,'Percent(%)', [round(x/df.shape[0]*100,2) for x in is_nan.Number_of_NaN])
is_nan.style.applymap(highlight_cells)


# In[7]:


# Overview nan value in credit_limit column
df[df['CREDIT_LIMIT'].isna()]


# In[8]:


# Drop row with nan value in credit_limit
df = df.drop(df[df.CREDIT_LIMIT.isna()].index, axis=0)


# In[9]:


# Show detaile of datapoint with missing values
desc_cust = df[df.MINIMUM_PAYMENTS.isna()].describe().T

desc_cust_df = pd.DataFrame(index=desc_cust.index, columns=desc_cust.columns, data=desc_cust )

f,ax = plt.subplots(figsize=(12,12),dpi=500)

sns.heatmap(desc_cust_df, annot=True,cmap = "Reds", fmt= '.0f',
            ax=ax,linewidths = 5, cbar = False,
            annot_kws={"size": 16})

ax.xaxis.set_ticks_position('top')
plt.tick_params(left=False, top=False)
plt.xticks(size = 18)
plt.yticks(size = 18)

plt.title("Descriptive Statistics of Missing Values", pad=30, x=0.31, y=1.02)
plt.show()


# ### Exploratory Data Analysis (EDA)

# In[10]:


# Overview correlation between features
plt.figure(figsize=(13,8), dpi=500)
sns.heatmap(round(df.corr(),2), annot=True, cmap='gist_heat_r')
plt.title("Correlation Between Features", pad=30)
plt.show()


# Since there is not much linear correlation between minimum_payments and other features, it is not possible to get the missing values from other features with the help of linear regression, so we use KNNimputer to fill the missing values.

# In[11]:


# Fill the remaining missing values with KNNimputer()
# Define imputer
imputer = KNNImputer()
# Fit on the dataset
imputer.fit(df)
# Transform the dataset
df1 = pd.DataFrame(imputer.transform(df),columns=df.columns)

# Print result
print(f'Missing (before): {df.isna().sum().sum()}')
print(f'Missing (after): {df1.isna().sum().sum()}')   


# In[12]:


# Check duplicated data
print(f"Number of dupilcated data: {df1.duplicated().sum()}" )


# In[13]:


# Draw pair plot for check noise and
pplot = sns.pairplot(df1, palette="#B41B10")
pplot.fig.suptitle("Pariplot of Features")
pplot.tight_layout()
plt.show()


# In[14]:


# num_plots = len(set(df1.columns) - set(['tenure']))
# fig, ax = plt.subplots((num_plots + 1) // 2, 2, figsize=(20, 5 * ((num_plots + 1) // 2)))
#
# for i, col in enumerate(set(df1.columns) - set(['tenure'])):
#     sns.boxplot(data=df1, x=col, palette="Reds_r", saturation=1, linewidth=2, ax=ax[i//2, i%2])
#
# plt.suptitle("Check Outlier with boxplot", y=1)
# plt.tight_layout(pad=3.0)
# plt.show()


# In[15]:


# # Check distribution of features
# fig, ax = plt.subplots(17, 1, figsize=(10,50))
# for i, col in enumerate(df1):
#     sns.histplot(df1[col], kde=True, ax=ax[i], color='red')
#
# fig.suptitle('Distribution of Columns',y=1.002)
# fig.tight_layout()
# plt.show()


# In[16]:


# Check distribution of some features in more detail
cols = ['PURCHASES', 'ONEOFF_PURCHASES', 'INSTALLMENTS_PURCHASES', 'CASH_ADVANCE', 'MINIMUM_PAYMENTS']
fig, ax = plt.subplots(5, 1, figsize=(10,25))
for i, col in enumerate(cols):
    sns.histplot(df1[df1 < 4000][col], kde=True, ax=ax[i], color='red')
    
fig.suptitle('Distribution of Some Columns (in more detail)',y=1.002)
fig.tight_layout(pad=3)
plt.show()


# In[17]:


# Check distribution of purchases_frequency and purchases_installment_frequency in more detail
fig, ax = plt.subplots(1,2,figsize=(15,5), dpi=100)
for i,col in enumerate(['PURCHASES_FREQUENCY', 'PURCHASES_INSTALLMENTS_FREQUENCY']):
    sns.histplot(df1[col], kde=True, color='red', ax=ax[i])

fig.suptitle('Distribution of Some Columns (in more detail)')
fig.tight_layout()
plt.show()


# - 👉 The purchases_frequency distribution has two peaks, one at zero (left side of the plot) and one at one (right side of the plot). Therefore, based on this feature, the data can be divided into two clusters:
# customers with low purchases_frequency
# customers with high purchases_frequency
# - 👉 The purchases_installments_frequency distribution also has two peaks, one at zero (left side of the plot) and one at one (right side of the plot). Therefore, based on this feature, the data can be divided into two clusters:
# customers with low purchases_installments_frequency
# customers with high purchases_installments_frequency

# In[18]:


# Correlation between features
plt.figure(figsize=(13,8), dpi=500)
sns.heatmap(round(df.corr(),2), annot=True, cmap='gist_heat_r')
plt.title("Correlation Between Features", pad=30)
plt.tight_layout()
plt.show()


# 👉 As can be seen, each customer's balance is more correlated with cash_advance and credit_limit. Also, the purchases of each customer have the highest correlation with oneoff_purchases, purchases_trx, installments_purchases and payments.

# ### Scale the data

# In[19]:


# Standardize dataset
scaler = StandardScaler().fit(df1)
df_trans = scaler.transform(df1)
df2 = pd.DataFrame(df_trans, columns = df1.columns)
df2


# ## DBSCAN Clustering

# In[20]:


# Using NearestNeighbors fucntion for to guess the right value for eps
knn = NearestNeighbors(n_neighbors = 4)
model = knn.fit(df2)
distances, indices = knn.kneighbors(df2)
distances = np.sort(distances, axis=0)
distances = distances[:,1]
plt.grid()
plt.plot(distances);
plt.xlabel('Points Sorted by Distance')
plt.ylabel('7-NN Distance')
plt.title('K-Distance Graph')


# In[21]:


# Finding best value for eps and min_sample
eps_values = np.arange(1,15,2) # eps values to be investigated
min_samples = np.arange(20,41,10) # min_samples values to be investigated

DBSCAN_params = list(product(eps_values, min_samples))


# Because DBSCAN creates clusters itself based on those two parameters let's check the number of generated clusters and evaluate reuslts by silhouette and calinski coefficient to tuning eps and min_samples.

# In[22]:


# Check optimom n_clusters for implement birch by using silhouette and calinski coefficient
silhouette_coef = []
num_of_clusters = []
for param in DBSCAN_params:
    dbscan = DBSCAN(eps=param[0], min_samples=param[1])
    dbscan.fit(df2)
    score = silhouette_score(df2, dbscan.labels_)
    silhouette_coef.append(score)
    # for check number of clusters
    num_of_clusters.append(len(np.unique(dbscan.labels_)))

calinski_harabasz_coef = []
for param in DBSCAN_params:
    dbscan = DBSCAN(eps=param[0], min_samples=param[1])
    dbscan.fit(df2)
    score = calinski_harabasz_score(df2, dbscan.labels_)
    calinski_harabasz_coef.append(score)


# In[23]:


# Check cluster results by heatmap
tmp = pd.DataFrame.from_records(DBSCAN_params, columns =['Eps', 'Min_samples'])   
tmp['No_of_clusters'] = num_of_clusters

pivot_1 = pd.pivot_table(tmp, values='No_of_clusters', index='Min_samples', columns='Eps')

fig, ax = plt.subplots(figsize=(12,6))
sns.heatmap(pivot_1, annot=True,annot_kws={"size": 16}, cmap="YlGnBu", ax=ax)
ax.set_title('Number of clusters')
plt.show()


# In[24]:


# Check silhouette score by heatmap
tmp = pd.DataFrame.from_records(DBSCAN_params, columns =['Eps', 'Min_samples'])   
tmp['Sil_score'] = silhouette_coef

pivot_1 = pd.pivot_table(tmp, values='Sil_score', index='Min_samples', columns='Eps')

fig, ax = plt.subplots(figsize=(18,6))
sns.heatmap(pivot_1, annot=True, annot_kws={"size": 10}, cmap="YlGnBu", ax=ax)
plt.show()


# In[25]:


# Check Calinski score by heatmap
tmp = pd.DataFrame.from_records(DBSCAN_params, columns =['Eps', 'Min_samples'])   
tmp['CH_score'] = calinski_harabasz_coef

pivot_1 = pd.pivot_table(tmp, values='CH_score', index='Min_samples', columns='Eps')

# fig, ax = plt.subplots(figsize=(18,6))
# sns.heatmap(pivot_1, annot=True, annot_kws={"size": 10}, cmap="YlGnBu", ax=ax)
# plt.show()


# In[26]:


import matplotlib.pyplot as plt
#
# def plot_evaluation(silhouette_scores, calinski_harabasz_scores, model_name, x):
#     plt.figure(figsize=(14, 6))
#
#     # Plot Silhouette Score
#     plt.subplot(1, 2, 1)
#     plt.plot(x, silhouette_scores, marker='o')
#     plt.title(f'Silhouette Score for {model_name}')
#     plt.xlabel('Parameter Index')
#     plt.ylabel('Silhouette Score')
#
#     # Plot Calinski-Harabasz Score
#     plt.subplot(1, 2, 2)
#     plt.plot(x, calinski_harabasz_scores, marker='o')
#     plt.title(f'Calinski-Harabasz Score for {model_name}')
#     plt.xlabel('Parameter Index')
#     plt.ylabel('Calinski-Harabasz Score')
#
#     plt.tight_layout()
#     plt.show()

# Example usage
# plot_evaluation(silhouette_coef, calinski_harabasz_coef, 'DBSCAN', x=range(len(DBSCAN_params)))


# In[27]:


# Coordinates of point in above table
for i in enumerate(DBSCAN_params):
    print(i)


# According to the above results, we consider eps=3.5 and min_samples=40 and implement the DBSCAN algorithm.

# In[28]:


# Implement DBSCAN for best eps and min_sample obtained from above
dbscan = DBSCAN(eps=3.5, min_samples=40)
dbscan.fit(df2)
# Store result of DBSCAN
pred = dbscan.labels_
# Number of cluster
num_of_clusters = len(np.unique(pred))
# Evaluate results of mean shift by silhouette and calinski coefficient
sh_score = silhouette_score(df2, pred)
ch_score = calinski_harabasz_score(df2, pred)
# Print results
print(f"Number of cluster: {np.unique(pred)}")
print(f"Silhouette Coefficient: {sh_score:.2f}")
print(f"Calinski Harabasz Coefficient: {ch_score:.2f}")


# In[29]:


best_model = pd.DataFrame(columns=['Model', 'Silhouette Score', 'Calinski-Harabasz Score', 'Parameters'])

# Example data
# df2 = ...  # Your dataset
# pred = ...  # Your DBSCAN predictions

# Store results obtained from DBSCAN
best_model.loc[len(best_model.index)] = [
    "DBSCAN",
    silhouette_score(df2, pred),
    calinski_harabasz_score(df2, pred),
    {"eps": 3.5, "min_samples": 40}
]


# In[30]:


# Adding the clusters column to the main dataframe (df2) and store in new dataframe
df_result_dbscan = pd.concat([df1, pd.DataFrame(pred, columns=['cluster'])], axis = 1)
df_result_dbscan


# In[31]:


# Initialize DBSCAN with appropriate parameters
dbscan = DBSCAN(eps=3.5, min_samples=40)

# Fit the model
dbscan.fit(df2)

# Store result of DBSCAN
pred_dbscan = dbscan.labels_

# Number of clusters
num_of_clusters = len(np.unique(pred_dbscan))

# Print results
print(f"Number of clusters: {num_of_clusters}")

# Store results in DataFrame
best_model = pd.DataFrame(columns=['Model', 'Silhouette Score', 'Calinski-Harabasz Score', 'Parameters'])
best_model.loc[len(best_model.index)] = [
    "DBSCAN",
    silhouette_score(df2, pred_dbscan),
    calinski_harabasz_score(df2, pred_dbscan),
    {"eps": 3.5, "min_samples": 40}
]

print(best_model)


# In[44]:


# Add the cluster labels to the DataFrame
df2['Cluster'] = dbscan.labels_

# Plot the clusters
plt.figure(figsize=(10, 6))
sns.scatterplot(data=df2, palette='viridis')
plt.title('DBSCAN Clustering ')
plt.show()

df2.drop(columns='Cluster',inplace=True)
# ## PCA (Principal Component Analysis)

# In[34]:


# Perform PCA
pca = PCA(n_components=2)  # You can choose the number of components
principal_components = pca.fit_transform(df2)

# Create a DataFrame with the principal components
df2_pca = pd.DataFrame(data=principal_components, columns=['PC1', 'PC2'])

# Add the principal components to the original DataFrame
pca_df2 = pd.concat([df2, df2_pca], axis=1)

print(pca_df2.head())


# In[35]:


pca.components_


# In[36]:


# Variance Ratio

pca.explained_variance_ratio_


# In[37]:


# Variance Ratio bar plot for each PCA components.
plt.figure(figsize = (10, 5))
ax = plt.bar(range(1,len(pca.explained_variance_ratio_)+1), pca.explained_variance_ratio_)
plt.xlabel("PCA Components",fontweight = 'bold')
plt.ylabel("Variance Ratio",fontweight = 'bold')

plt.show()


# In[38]:


# Check the column names
print(pca_df2.columns)


# In[45]:


plt.figure(figsize=(20, 5))

# Plot PC1 vs PC2
plt.subplot(1, 2, 1)
sns.scatterplot(data=pca_df2, x='PC1', y='PC2')
plt.title('PC1 vs PC2')


# ## DBSCAN Clustering on PCA-Transformed Data'

# In[40]:


dbscan__ = DBSCAN(eps=3.5, min_samples=40)
dbscan__.fit(pca_df2)

# Add the cluster labels to the DataFrame
pca_df2['Cluster'] = dbscan__.labels_

# Plot the clusters
# plt.figure(figsize=(10, 6))
# sns.scatterplot(data=pca_df2, x='PC1', y='PC2', hue='Cluster', palette='viridis')
# plt.title('DBSCAN Clustering on PCA-Transformed Data')
# plt.show()


# ### Clustering Result

# - Cluster -1: Customers with average balance and low credit_limit that frequently purchase
# - Cluster 0: Cutomers with low balance and low credit limit that they are not frequently purchase in type of oneoff or installment.


import pickle

import pickle

# Assuming you have trained your model and other necessary components
# For example:
dbscan_model = dbscan__  # Your trained DBSCAN model
scaler_model = scaler  # Your StandardScaler object

# Save the trained models and other necessary components using pickle
with open('dbscan_model.pkl', 'wb') as file:
    pickle.dump(dbscan_model, file)

with open('scaler_model.pkl', 'wb') as file:
    pickle.dump(scaler_model, file)

# If you have other components like PCA, you can save them similarly:
with open('pca_model.pkl', 'wb') as file:
    pickle.dump(pca, file)

with open('df_csv_file.pkl', 'wb') as file:
    pickle.dump(df_result_dbscan, file)