import cv2
import matplotlib.pyplot as plt
import numpy as np

from sklearn.cluster import KMeans


#%% Working with images:
# Loading image:
image = cv2.imread('foto1.jpeg') #nombre del archivo de la imagen
print(f"image.shape: {image.shape}")

# Changing image from BGR -> RGB:
rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

#%% IA Stage:
k = 3
model = KMeans(n_clusters=k, random_state=0)

# data wrangling:
x = rgb_image.reshape((-1, 3))
print(f"x.shape = {x.shape}")

# model training:
model.fit(x)

# Labels:
labels = model.labels_
print(f"labels = {labels}")

#%% Compressed image visualization:
centroids = np.uint8(model.cluster_centers_) # To define colors of the compressed image.
print(f"centroids = {centroids}")

# Segmented image:
segmented_image = centroids[labels.flatten()]

# Adjusting dimensions:
segmented_image = segmented_image.reshape(rgb_image.shape)

# Plotting:
# plt.figure(dpi=500)
plt.subplot(2, 2, 1)
plt.imshow(rgb_image)
plt.axis('off')
plt.title('Original Image')

plt.subplot(1, 2, 2)
plt.imshow(segmented_image)
plt.axis('off')
plt.title('Segmented image')
plt.show()

#%% Eliminación de objeto:
new_labels = []
for label in labels:
    if label == 2:
        new_labels.append(0)
    else:
        new_labels.append(label)

new_labels = np.array(new_labels)
# Segmented image:
segmented_image = centroids[new_labels.flatten()]

# Adjusting dimensions:
segmented_image = segmented_image.reshape(rgb_image.shape)

# Plotting:
# plt.figure(dpi=500)
plt.subplot(1, 2, 1)
plt.imshow(rgb_image)
plt.axis('off')
plt.title('Original Image')

plt.subplot(1, 2, 2)
plt.imshow(segmented_image)
plt.axis('off')
plt.title('Segmented image')
plt.show()