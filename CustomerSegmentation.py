# K-Means Clustering
# It is used to separate raw data into groups.
# We should not categorize raw data. AI does the classification.

from sklearn.cluster import KMeans
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from matplotlib import pyplot as plt

df = pd.read_csv('Avm_Customers.csv')

plt.scatter(df['Annual Income (k$)'], df['Spending Score (1-100)'])
plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending Score (1-100)')
plt.show

# Renaming some columns because some of them long 
df.rename(columns={'Annual Income (k$)' : 'Income'}, inplace = True)
df.rename(columns={'Spending Score (1-100)' : 'Score'}, inplace = True)

# Normalization
scaler = MinMaxScaler()
scaler.fit(df[['Income']])
df['Income'] = scaler.transform(df[['Income']])

scaler.fit(df[['Score']])
df['Score'] = scaler.transform(df[['Score']])
df

# Determining K value by Elbow Method
k_range = range(1,11)

list_dist = []

for k in k_range:
    kmeans_model = KMeans(n_clusters = k)
    kmeans_model.fit(df[['Income', 'Score']])
    list_dist.append(kmeans_model.inertia_)

plt.xlabel('K')
plt.ylabel('Distortion Value (Inertia)')
plt.plot(k_range, list_dist)
plt.show()

# The best K value is 5

kmeans_model = KMeans(n_clusters = 5)
y_predicted = kmeans_model.fit_predict(df[['Income', 'Score']])
y_predicted

df['cluster'] = y_predicted
df

# Show centroids
kmeans_model.cluster_centers_

df1 = df[df.cluster == 0]
df2 = df[df.cluster == 1]
df3 = df[df.cluster == 2]
df4 = df[df.cluster == 3]
df5 = df[df.cluster == 4]

plt.xlabel('Income')
plt.ylabel('Score')

plt.scatter(df1['Income'], df1['Score'], color = 'green')
plt.scatter(df2['Income'], df2['Score'], color = 'red')
plt.scatter(df3['Income'], df3['Score'], color = 'black')
plt.scatter(df4['Income'], df4['Score'], color = 'orange')
plt.scatter(df5['Income'], df5['Score'], color = 'purple')

plt.scatter(kmeans_model.cluster_centers_[:,0], kmeans_model.cluster_centers_[:,1], color='blue', marker='X', label="centroids")
plt.legend()
plt.show()




