import numpy as np
import matplotlib.pyplot as plt
import feature_utils as utils
import time
from sklearn.naive_bayes import GaussianNB
from sklearn.cross_validation import StratifiedKFold

f_name = 'hog_9'
X, labels = utils.load_features(f_name)
n_samples, n_features = X.shape
n_targets = labels.shape[1]
#n_targets = 1
print(X.shape)

n_folds = 10
accuracy = np.ndarray((n_folds, n_targets))
tpr = np.ndarray((n_folds, n_targets))
fpr = np.ndarray((n_folds, n_targets))


for target in range(n_targets):
	print("target : %i/%i" % (target+1, n_targets))
	y = labels[:, target]
	skf = StratifiedKFold(y, n_folds=n_folds)
	
	start_time = time.time()
	for i, (train, test) in enumerate(skf):
		print("fold : %i/%i" % (i+1, n_folds))
		
		classifier = GaussianNB()
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

np.save("results/%s_gaussianNB_accuracy" % f_name, accuracy)
np.save("results/%s_gaussianNB_tpr" % f_name, tpr)
np.save("results/%s_gaussianNB_fpr" % f_name, fpr)

plt.plot(np.mean(accuracy, axis=0), label="accuracy")
plt.plot(np.mean(tpr, axis=0), label="tpr")
plt.plot(np.mean(fpr, axis=0), label="fpr")
plt.legend(loc="lower right")
plt.show()
