#!/usr/bin/env python
# coding: utf-8


"""
This Project is based on Customer Segmentation using K-Means Clustering. 
Customers will be partitioned into k number of clusters, each cluster will contain customers with similar characteristics. 
"""



# Importing Libraries to be used in this Project

import pandas as pd
import seaborn as sns

import matplotlib.pyplot as plt
import plotly.express as px
from scipy.stats.mstats import trimmed_var
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from urllib.request import urlretrieve


# Data Preparation
    # Importation of Data (Flat files) from IBM storage (web) and load into a Dataframe
    # Exploration of imported data

url ='https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-ML0101EN-SkillsNetwork/labs/Module%204/data/Cust_Segmentation.csv'  
urlretrieve(url, 'Cust_Segmentation.csv') # Using 'urlretrieve' function saves your imported file locally
df = pd.read_csv("Cust_Segmentation.csv", sep = ",", index_col= "Customer Id")
print(df.shape)
df.head()



# Data Preparation
    # Exploration of imported data
    
# Applying 'info' method on the Dataframe, the data types of each feaures are known including the ones with 'NaN' values. 
df.info()

# Applyng 'isnull' and 'sum' method calculates the number of 'NaN' values in each feature (column)
df.isnull().sum()

# Applying 'describe' method shows the summary properties of each feature
df.drop(columns = "Address").describe()



# Further Data Exploration, Boxplot of 'Income' feature using seaborn
sns.boxplot(
    x = df["Income"],
    orient= "h",
)
plt.title("Box plot of Income earned");




# Data Cleanng, preparation, amd Splitting
    # Dropped "Address" column because its a categorical feature 
    # "Defaulted" column tend to be the only column with 'NaN' values.




# Calulates the Variance of each feature to determine the most significant features for the clustering
    # Using Normal variance calculations
df.drop(columns = "Address").var().sort_values()




# Calulates the Variance of each feature to determine the most significant features for the clustering
    # Using trimmed variance calculations should in case there are Outliers
df.drop(columns = "Address").apply(trimmed_var, limits = (0.1, 0.1)).sort_values()   



# From the above variance (normal and trimmed) calcultions, "Defaulted" and "Income"column has the least and highest variance.
# I will drop "Defaulted" column because it has least and most Insignficant value of variance




# Visualisation (Bar chart representation) of the features based on Variance
fig = px.bar(
    df.drop(columns = ["Address", "Defaulted"]).var().sort_values(),
    title = "Bar Chart of High (Relevant) variance feature",
    orientation="h"
)
fig.update_layout(xaxis_title = "Variance Value", yaxis_title = "High Variance Features")
fig.show()




# Visualisation (Bar chart representation) of the features based on Variance(trimmed)
fig = px.bar(
    df.drop(columns = ["Address", "Defaulted"]).apply(trimmed_var).sort_values(),
    title = "Bar Chart of High (Relevant) variance feature (Trimmed)",
    orientation="h"
)
fig.update_layout(xaxis_title = "Variance Value", yaxis_title = "High Variance Features (Trimmed)")
fig.show()




# Selecting Data (X) for the K-Means Clustering
X = df.drop(columns = ["Address", "Defaulted"])




# Normalization of Seleted Data and Model building
# Before Normalization of Data and the Building of Model, the rght value of "K" (Number of Clusters) has to be determined by plotting "interia" and "silhouette score" against "Number of Clusters"




n_clusters = range(2, 13) # Number of clusters
inertia_errors = []
silhouette_scores = list()

# Add `for` loop to train model and calculate inertia, silhouette score.
for k in n_clusters:
    # Build and fit a  Model 
    model = make_pipeline(
        StandardScaler(),
        KMeans(init="k-means++", n_clusters=k, n_init=12, random_state=42)
    )
    model.fit(X)
    # Calculate and append Inertia to inertia_errors
    inertia_errors.append(model.named_steps["kmeans"].inertia_)
    # Calculate and append silhouette_score to silhouette_scores
    silhouette_scores.append(silhouette_score(X, model.named_steps["kmeans"].labels_))

print("Inertia:", inertia_errors[:10])
print()
print("Silhouette Scores:", silhouette_scores[:4])




# Create line plot of `inertia_errors` vs `n_clusters`
fig = px.line(
    x = n_clusters,
    y = inertia_errors,
    title = "K-Means Model: Inertia vs Number of Clusters"
)
fig.update_layout(xaxis_title = "Number of Clusters", yaxis_title = "Inertia")

fig.show()




# Create a line plot of `silhouette_scores` vs `n_clusters`
fig = px.line(
    x = n_clusters,
    y = silhouette_scores,
    title = "K-Means Model: Silhouette Score vs Number of Clusters"
)
fig.update_layout(xaxis_title = "Number of Clusters", yaxis_title = "Silhouette Score")

fig.show()




# By comparing the graph of "interia" against "Number of Clusters" and "silhouette score" against "Number of Clusters".
# One can conclude that the rght value of "K" is 3




# Building the final model
model_finale = make_pipeline(
    StandardScaler(),
    KMeans(init="k-means++", n_init=10, n_clusters=3, random_state=42)
)
model_finale.fit(X)




# Accessing the label from the Model (model_finale)
labels = (model_finale.named_steps["kmeans"].labels_)
Xav = X.groupby(labels).mean()
Xav



# Create side-by-side bar chart of `Xav`
fig = px.bar(
    Xav, barmode = "group",
    #labels= labels,
    title = "Customer segmentation Using KMeans Clustering"
)
fig.update_layout(xaxis_title = "Clusters", yaxis_title = "Value of selected features")

fig.show()




# A Scatter plot of 'Income' Vs 'Age' using Matplotlib; the size of each dot is determine by level of Education 
import numpy as np

area = np.pi * ( X["Edu"])**2  
plt.scatter(X["Age"], X["Income"], s=area, c=labels.astype(float), alpha=0.5)
plt.title("A Graph of Income against Age")
plt.xlabel('Age', fontsize=18)
plt.ylabel('Income', fontsize=16)

plt.show()




# A Scatter plot of 'Income' Vs 'Age' using Plotly; the size of each dot is determine by level of Education 
fig = px.scatter(
    x = X["Age"], 
    y = X["Income"],
    size = np.pi * ( X["Edu"])**2,
    color = labels.astype(str),
    title = "A Graph of Income against Age"
)
fig.update_layout(xaxis_title = "Age", yaxis_title = "Income")

fig.show()





# 3D repesenttion of 'Education', 'Age', and 'Income'
from mpl_toolkits.mplot3d import Axes3D 
fig = plt.figure(1, figsize=(8, 6))
plt.clf()
ax = Axes3D(fig, rect=[0, 0, .95, 1], elev=48, azim=134)

plt.cla()
# plt.ylabel('Age', fontsize=18)
# plt.xlabel('Income', fontsize=16)
# plt.zlabel('Education', fontsize=16)
ax.set_xlabel('Education')
ax.set_ylabel('Age')
ax.set_zlabel('Income')

ax.scatter(X["Edu"], X["Age"], X["Income"], c= labels.astype(float));





"""
# Based on the insights gotten from the various visualizations above, one can say that there are customers with;

    #1. Low Income, average education and young (cluster 0)
    #2. Middle Income, Mostly low education, middle aged but older the rest class (clusters 1)
    #3. High Income (Rich), highly educated, and middle age(cluster 2)
"""    


