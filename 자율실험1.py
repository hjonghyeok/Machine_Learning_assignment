# 자율실험 1번

import numpy as np
import skimage.io as io
from matplotlib import pyplot as plt
from sklearn import cluster
import time

file = './road_sign_gray.jpg'
img = io.imread(file)

im = np.array(img)
x = im.reshape(-1,im.size).T

Times = []
k_s = []

for k in range(2,20):
    times = []
    for i in range(0,10):
        start = time.time()
        kmeans = cluster.KMeans(n_clusters=k)
        kmeans.fit(x)
        end = time.time()
        times.append(end-start)
    k_s.append(k)
    Times.append(np.mean(times))


plt.plot(k_s,Times)
plt.xlabel("k")
plt.ylabel("Time")
plt.show()