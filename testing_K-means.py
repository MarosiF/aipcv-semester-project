import numpy as np
import matplotlib.pyplot as plt

import feature_utils as utils
import time
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.cross_validation import KFold
from multiprocessing.dummy import Pool
from multiprocessing import cpu_count

X, labels = utils.load_features("gist")
n_samples, n_features = X.shape
n_targets = labels.shape[1]
#n_targets = 1
print(X.shape)

n_folds = 4

# runs for each fold
def run(args):
	start_time = time.time()
	train, test = args

	# for GIST only
	mult = 32  # split each input row into 32 rows
	fv_dims = 512 / mult
	X_train = X[train].reshape((-1, fv_dims))

	# cluster data
	kmeans = KMeans()
	kmeans.fit(X_train)
	assigned_centers = kmeans.predict(X_train)
	# for each cluster center (row) and each attribute (col)
	# gives a truth value indicating class membership
	center_attribute_assignment = np.ndarray((kmeans.n_clusters, n_targets), dtype=bool)


	for attr in range(n_targets):
		# print("attribute : %i/%i" % (attr+1, n_targets))

		# find whether a cluster belongs to attr
		y = labels[train, attr].repeat(mult)

		center_yes = np.ndarray((kmeans.n_clusters,), dtype=int)
		center_no = np.ndarray((kmeans.n_clusters,), dtype=int)
		for i, center in enumerate(assigned_centers):
			if y[i]:
				center_yes[center] += 1
			else:
				center_no[center] += 1

		center_attribute_assignment[:, attr] = center_yes / (center_yes + center_no)

	acc = np.ndarray((0, n_targets))
	tpr = np.ndarray((0, n_targets))
	fpr = np.ndarray((0, n_targets))

	# classify
	for img, label in zip(X[test], labels[test]):
		X_test = img.reshape((-1, fv_dims))
		y = label.reshape((1, n_targets)).repeat(mult, axis=0)
		pred_cc = kmeans.predict(X_test)
		assignments = center_attribute_assignment[pred_cc]
		assert y.shape == assignments.shape, "{} vs {}".format(y.shape, assignments.shape)
		acc = np.vstack((acc, np.sum(y == assignments, axis=0) / mult))
		tpr = np.vstack((tpr, np.sum(y & assignments, axis=0) / mult))
		fpr = np.vstack((fpr, np.sum(~y & assignments, axis=0) / mult))

	acc = np.average(acc, axis=0)
	tpr = np.average(tpr, axis=0)
	fpr = np.average(fpr, axis=0)

	elapsed_time = time.time() - start_time
	print("elapsed time : %.4f" % elapsed_time)

	return acc, tpr, fpr

skf = KFold(len(X), n_folds=n_folds)

accuracy = np.ndarray((n_folds, n_targets))
tpr = np.ndarray((n_folds, n_targets))
fpr = np.ndarray((n_folds, n_targets))

with Pool(cpu_count()) as pool:
	for i, (a, t, f) in enumerate(pool.imap(run, skf)):
		accuracy[i] = a
		tpr[i] = t
		fpr[i] = f

np.save("results/gist_K-means_accuracy", accuracy)
np.save("results/gist_K-means_tpr", tpr)
np.save("results/gist_K-means_fpr", fpr)

plt.plot(np.mean(accuracy, axis=0), label="accuracy")
plt.plot(np.mean(tpr, axis=0), label="tpr")
plt.plot(np.mean(fpr, axis=0), label="fpr")
plt.legend(loc="lower right")
plt.savefig("results/gist_K-means.png")
plt.show()
