#sklearn 라이브러리를 이용하여 step3의 박스를 완성하라. step1,2,4박스는 제시된 코드를 이용, 다른 방법으로 구현도 가능

import numpy as np
import skimage.io as io
from matplotlib import pyplot as plt
from sklearn import cluster
import time

# step 1
file = './road_sign_gray.jpg'
img = io.imread(file)
io.imshow(img)
plt.show()

# step 2
im = np.array(img)
x = im.reshape(-1,im.size).T

# step 3
k = 2   # k = 2, 3, 4, 5, 6, 7, 8, 9, 10 ... 

model = cluster.KMeans(n_clusters=k)
model.fit(x)
labels = model.predict(x)

# step 4
kimg= labels.reshape(im.shape[0], im.shape[1]) 
io.imshow(kimg,cmap="gray")
plt.show()