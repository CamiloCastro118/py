import pandas as pd
import numpy as np

from sklearn.cluster import SpectralClustering
from sklearn.metrics import davies_bouldin_score
from matplotlib import pyplot as plt


#%% Load dataset:
dataset = pd.read_csv('spirals.csv')
print(dataset)

dataset = dataset.drop(['color'], axis=1)

# Plotting dataset:
# plt.scatter(dataset['x'], dataset['y'])
# plt.title('Dataset')
# plt.xlabel('x')
# plt.ylabel('y')
# plt.show()

x = np.array(dataset)

#%% Model training:
k = 5

spectral_model = SpectralClustering(n_clusters=k, affinity='nearest_neighbors', n_neighbors=40, random_state=0)
spectral_model.fit(x)

spectral_labels = spectral_model.labels_
# print(f'spectral_labels = {spectral_labels}')

# Plotting the labels:
colors = ['red', 'green', 'blue', 'yellow', 'cyan', 'black']
plt.figure(dpi=300)
for label in np.unique(spectral_labels):
    plt.scatter(
        dataset['x'][spectral_labels == label],
        dataset['y'][spectral_labels == label],
        c=colors[label],
        label='Cluster ' + str(label)
    )

plt.title('Davies-Bouldin')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.show()

#%% Metrics:
score = davies_bouldin_score(x, spectral_labels)
print(f'Davies Bouldin score: {round(score, 2)}')