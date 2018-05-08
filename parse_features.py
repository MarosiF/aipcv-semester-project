import numpy as np
import matplotlib.pyplot as plt
import feature_utils as utils
import scipy.io as sio
import os


# Parameters in order to parse mat files into npy files
f_name = 'hog_9'		# feature name
feature_name = 'hog' 	# feature name in the mat file
n_features = 576		# number of features (needs to be specified)
n = 1 					# number of individual samples per sample (only needed for histograms_2x2, histograms_4x4)

folder_features = "../features/%s" % f_name
output_path = "../features/%s.npy" % f_name

attributes = utils.attributes()
histogramsPathList = os.listdir(folder_features)
n_samples = len(histogramsPathList)
indexes = np.ndarray((n_samples,))
if n > 1: 	X = np.ndarray((n_samples, n_features, n))
else: 		X = np.ndarray((n_samples, n_features))

for i, histogramsPath in enumerate(histogramsPathList):
	if i % 1000 == 0: print("%i/%i" % (i, n_samples))
	path = os.path.join(folder_features, histogramsPath)
	
	fd = sio.loadmat(path)[feature_name]
	if n > 1:	X[i] = fd
	else:		X[i] = fd[0]

	idx = int(os.path.splitext(histogramsPath)[0])
	indexes[i] = idx

indexes = np.array(indexes)
idx = np.argsort(indexes)

X = X[idx, :]
np.save(output_path, X)
print(X.shape)