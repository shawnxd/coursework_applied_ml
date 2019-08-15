# coding: utf-8

# In[1]:


import sys

k, inputImageFilename, outputImageFilename = sys.argv[1], sys.argv[2], sys.argv[3]
k = int(k)


# In[2]:


import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from PIL import Image
# k = 3
# import random
# from sklearn.cluster import KMeans
# from sklearn import preprocessing
# from sklearn.decomposition import PCA, NMF


# In[3]:


# Importing and Reshaping the Data
# inputImageFilename = 'image_2.jpg'
pic_3D_arr = plt.imread(inputImageFilename).astype(int)


# print(pic_3D_arr.shape)
# k = 3


# In[4]:


def standardize(X):
    sd = X.std(axis=0)
    mean_v = X.mean(axis=0)
    X_std = np.where(sd != 0, (X - mean_v) / sd, 0)
    return X_std


# In[5]:


def kMeans(X, k):
    m = int(X.shape[0])
    cluster = np.mat(np.zeros((m, 2)))
    n = int(X.shape[1])
    centroids = np.mat(np.zeros((k, n)))
    for j in range(n):
        min_v = min(X[:, j])
        max_v = max(X[:, j])
        range_v = max_v - min_v
        centroids[:, j] = min_v + range_v * np.random.rand(k, 1)
    cluster_change = True
    while cluster_change:
        for i in range(m):
            minDist = np.inf
            minIndex = -1
            for j in range(k):
                distJI = np.sqrt(np.sum(np.power(centroids[j, :] - X[i, :], 2)))
                #                 print(distJI)
                #                 print(minDist)
                if distJI < minDist:
                    minDist = distJI
                    minIndex = j
            if cluster[i, 0] != minIndex:
                clusterChanged = True
            cluster[i, :] = minIndex, minDist ** 2
        # print(centroids)
        for cent in range(k):
            ptsInClust = X[np.nonzero(cluster[:, 0] == cent)[0]]
            if ptsInClust.shape[0] == 0:
                for j in range(n):
                    min_v = min(X[:, j])
                    max_v = max(X[:, j])
                    range_v = max_v - min_v
                    centroids[:, j] = min_v + range_v * np.random.rand(k, 1)
            else:
                centroids[cent, :] = np.mean(ptsInClust, axis=0)
        cluster_change = False
    cluster_num = []
    for i in cluster:
        #         print(i)
        cluster_num.append(int(i.tolist()[0][0]))
    return centroids, cluster_num


# In[ ]:


pic_5 = []
# print(len(pic_3D_arr))
for i in range(len(pic_3D_arr)):
    for j in range(len(pic_3D_arr[0])):
        pic_5.append([pic_3D_arr[i][j][0], pic_3D_arr[i][j][1], pic_3D_arr[i][j][2], i, j])
pic_5_arr = np.array(pic_5)
pic_5_fin = pic_5_arr
# print(pic_5_arr)


# In[ ]:


pic_s_std = standardize(pic_5_arr)
sd = pic_5_arr.std(axis=0)
mean_v = pic_5_arr.mean(axis=0)


# In[ ]:


centroids, cluster_lst = kMeans(pic_s_std, k)
# print(centroids)
# print(cluster)


# In[ ]:


for i in range(len(pic_5_fin)):
    #     print(cluster_lst[i])
    #     print(centroids[cluster_lst[i]].tolist()[0][0])
    pic_5_fin[i][0] = centroids[cluster_lst[i]].tolist()[0][0]
    pic_5_fin[i][1] = centroids[cluster_lst[i]].tolist()[0][1]
    pic_5_fin[i][2] = centroids[cluster_lst[i]].tolist()[0][2]
# print(pic_5_fin)
pic_5_fin = np.where(sd != 0, pic_5_fin * sd + mean_v, 0)
for i in range(len(pic_5_fin)):
    x = int(pic_5_arr[i].tolist()[3])
    y = int(pic_5_arr[i].tolist()[4])
    #     print(x)
    #     print(y)
    pic_3D_arr[x][y][0] = pic_5_fin[i][0]
    pic_3D_arr[x][y][1] = pic_5_fin[i][1]
    pic_3D_arr[x][y][2] = pic_5_fin[i][2]
# print(pic_3D_arr)

img = Image.fromarray(np.uint8(pic_3D_arr))
img.save(outputImageFilename)


# In[ ]:


# plt.imshow(pic_3D_arr)
