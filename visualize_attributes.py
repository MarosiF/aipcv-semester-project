import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio
import feature_utils as utils
import os

labels = sio.loadmat(utils.path_labels)['labels_cv']

print(labels.shape)
n_samples, n_attributes = labels.shape

x = np.unique(labels)
for i in range(n_attributes):
	y = np.histogram(labels[:,i], bins=4)[0].astype(float)
	y = 100*y/n_samples
	plt.plot(x, y, 'b', alpha=0.2)

plt.xlabel("Attribute Strength in Percent")
plt.ylabel("Occurence in Percent")
plt.show()