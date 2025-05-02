import pandas as pd
import numpy as np

from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from matplotlib import pyplot as plt


#%% Load dataset:
dataset = pd.read_csv('spirals.csv')
print(dataset)

dataset = dataset.drop(['color'], axis=1)

# Plotting dataset:
plt.scatter(dataset['x'], dataset['y'])
plt.title('Dataset')
plt.xlabel('x')
plt.ylabel('y')
plt.show()

x = np.array(dataset)

#%% Model training:
k = 5
kmeans_model = KMeans(n_clusters=k)
kmeans_model.fit(x)

kmeans_labels = kmeans_model.labels_
print(f'kmeans_labels = {kmeans_labels}')

# Plotting the labels:
colors = ['red', 'green', 'blue', 'yellow', 'cyan', 'black']
plt.figure(dpi=300)
for label in np.unique(kmeans_labels):
    plt.scatter(
        dataset['x'][kmeans_labels == label],
        dataset['y'][kmeans_labels == label],
        c=colors[label],
        label='Cluster ' + str(label)
    )

plt.title('Silhouette')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.show()

#%% Metrics:
score = silhouette_score(x, kmeans_labels)
print(f'Silhouette score: {round(score, 2)}')
