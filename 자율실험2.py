# 자율실험 2번

import numpy as np
import skimage.io as io
from matplotlib import pyplot as plt
from sklearn import cluster

file = './road_sign_gray.jpg'
img = io.imread(file)

im = np.array(img)
x = im.reshape(-1,im.size).T

Inertias = []
k_s = []

for k in range(2,21):
    inertias = []
    for i in range(10):
        kmeans = cluster.KMeans(n_clusters=k)
        kmeans.fit(x)
        inertias.append(kmeans.inertia_)
    k_s.append(k)
    Inertias.append(np.mean(inertias))

plt.plot(k_s,Inertias)
plt.xlabel("k")
plt.ylabel("Inertia")
plt.show()