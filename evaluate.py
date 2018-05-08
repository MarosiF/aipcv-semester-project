import numpy as np
import matplotlib.pyplot as plt
import feature_utils as utils
import time
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import GaussianNB
from sklearn.decomposition import PCA
from sklearn.cross_validation import StratifiedKFold
from sklearn.metrics import confusion_matrix

# Settings
classifiers = {'bayes' : GaussianNB(), 'SVM' : LinearSVC()}
features = ['hog_9', 'hog_18', 'gist', 'histogram']
f_name = 'hog_9'
c_name = 'bayes'
folder = 'results/confusion_evaluation'
n_folds = 10


# Load data
X, labels = utils.load_features(f_name)
X = X[:, :9] # First feature
n_samples, n_features = X.shape
n_targets = labels.shape[1]
print(X.shape)


# Evaluation data
evaluation = np.ndarray((n_targets, n_folds, 2, 2))

for target in range(n_targets):
	print("target : %i/%i" % (target+1, n_targets))
	y = labels[:, target]

	skf = StratifiedKFold(y, n_folds=n_folds)
	start_time = time.time()
	for i, (train, test) in enumerate(skf):
		print("fold : %i/%i" % (i+1, n_folds))

		classifier = classifiers[c_name]
		classifier.fit(X[train], y[train])
		classes = classifier.predict(X[test])

		evaluation[target, i, :, :] = confusion_matrix(y[test], classes)

	elapsed_time = time.time() - start_time
	print("elapsed time : %.4f" % elapsed_time)

np.save("%s/evaluation_%s_%s" % (folder, f_name, c_name), evaluation)
