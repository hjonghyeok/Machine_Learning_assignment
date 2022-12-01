# k를 2부터 증가하면서 군집화된 영상을 출력하라. 
# 클러스터 수(k)에 따른 실험결과

import numpy as np
import skimage.io as io
from matplotlib import pyplot as plt
from sklearn import cluster

file = './road_sign_gray.jpg'
img = io.imread(file)

im = np.array(img)
x = im.reshape(-1,im.size).T

for k in range(2,100):
    kmeans = cluster.KMeans(n_clusters=k)
    kmeans.fit(x)
    labels = kmeans.predict(x)
    kimg= labels.reshape(im.shape[0], im.shape[1]) 
    io.imshow(kimg,cmap="gray")
    plt.title("k = "+str(k))
    plt.show()