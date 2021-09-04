from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt
import numpy as np
import argparse

np.random.seed(1)

(X, y) = make_blobs(n_samples=1000, n_features=2, centers=2, cluster_std=1.5, random_state=1)
y = y.reshape((y.shape[0], 1))

X = np.c_[X, np.ones((X.shape[0]))]

(trainX, testX, trainY, testY) = train_test_split(X, y, test_size=0.5, random_state=42)

penalty = 0
W = np.random.randn(X.shape[1], 1)
print(f"W = {W}")
print(f'W.shape = {W.shape}')
print(f'W.shape[0] = {W.shape[0]}')
for i in np.arange(0, W.shape[0]):
    for j in np.arange(0, W.shape[1]):
        penalty += (W[i][j] ** 2)
        print(f"W[i][j] = {W[i][j]}")
        print(f"W[i][j ** 2] = {W[i][j] ** 2}")
        # print(f"W[i] = {W[i]}")
        # print(f"W[j] = {W[j]}")
print(penalty)