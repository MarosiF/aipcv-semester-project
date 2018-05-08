import numpy as np
import matplotlib.pyplot as plt
import feature_utils as utils
import time
from sklearn import neighbors, metrics
from sklearn.decomposition import PCA

X, y = utils.load_features('histograms')
n_samples, n_features = X.shape
y = y[:, 0]
print(X.shape)

pca = PCA(n_components=0.7)
X = pca.fit_transform(X)
print(X.shape)

start_time = time.time()

k = 5
classifier = neighbors.KNeighborsClassifier(k)
classifier.fit(X, y)
p = classifier.predict_proba(X)
idx = np.argmax(p, axis=1)
scores = p[np.arange(p.shape[0]), idx]
genuine = classifier.classes_[idx] == y

elapsed_time = time.time() - start_time

fpr, tpr, _ = metrics.roc_curve(genuine, scores)

print("elapsed time : " + str(elapsed_time))
print(metrics.auc(fpr, tpr))
print("%i/%i" % (np.sum(genuine), n_samples))

plt.plot(fpr, tpr, 'b')
plt.plot([0, 1], [0, 1], 'k')
plt.show()
