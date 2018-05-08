import numpy as np
import matplotlib.pyplot as plt
import feature_utils as utils
import time
from sklearn import neighbors, metrics
from sklearn.decomposition import PCA
from sklearn.cross_validation import StratifiedKFold

X, labels = utils.load_features('histograms')
n_samples, n_features = X.shape
n_targets = labels.shape[1]
#n_targets = 1
print(X.shape)

pca = PCA(n_components=0.95)
X = pca.fit_transform(X)
print(X.shape)
c = X.shape[1]

n_folds = 10
accuracy = np.ndarray((n_folds, n_targets))
tpr = np.ndarray((n_folds, n_targets))
fpr = np.ndarray((n_folds, n_targets))
k = 5

for target in range(n_targets):
	print("target : %i/%i" % (target+1, n_targets))
	y = labels[:, target]
	skf = StratifiedKFold(y, n_folds=n_folds)

	start_time = time.time()
	for i, (train, test) in enumerate(skf):
		print("fold : %i/%i" % (i+1, n_folds))
		
		classifier = neighbors.KNeighborsClassifier(k)
		classifier.fit(X[train], y[train])
		classes = classifier.predict(X[test])

		pp = np.sum(y[test]).astype(float)
		nn = np.sum(~y[test]).astype(float)
		tp = np.sum(classes & y[test]).astype(float)
		fp = np.sum(~classes & y[test]).astype(float)
		genuine = np.sum(classes == y[test]).astype(float)
		
		accuracy[i, target] = genuine/len(y[test])
		tpr[i, target] = tp/pp
		fpr[i, target] = fp/nn

	elapsed_time = time.time() - start_time
	print("elapsed time : %.4f" % elapsed_time)

np.save("results/histograms_pca%i_knn%i_accuracy" % (c, k), accuracy)
np.save("results/histograms_pca%i_knn%i_tpr" % (c, k), tpr)
np.save("results/histograms_pca%i_knn%i_fpr" % (c, k), fpr)

#plt.subplot(1, 2, 1)
plt.plot(np.mean(accuracy, axis=0), label="accuracy")
plt.plot(np.mean(tpr, axis=0), label="tpr")
plt.plot(np.mean(fpr, axis=0), label="fpr")
plt.legend(loc="lower right")
#plt.subplot(1, 2, 2)
#plt.imshow(tpr)
#plt.colorbar()
plt.show()