import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio
import feature_utils as utils
from sklearn.decomposition import PCA

labels = utils.labels() > 0.5

X = np.load("histograms.npy")
n_samples, n_features = X.shape


print(X.shape)

pca = PCA(n_components=2)
pca.fit(X)
X = pca.transform(X)
print(X.shape)

print(pca.explained_variance_ratio_)
print(len(pca.explained_variance_ratio_))



y = labels[:, 0]
plt.figure()
plt.scatter(X[np.invert(y), 0], X[np.invert(y), 1], color='b')
plt.scatter(X[y, 0], X[y, 1], color='r')

plt.figure()
plt.plot(pca.explained_variance_ratio_)
plt.show()