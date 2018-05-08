import numpy as np
import matplotlib.pyplot as plt
import feature_utils as utils
import os
from os import path

folder = 'results/confusion_evaluation'
files = os.listdir(folder)

for f in files:
	C = np.load(path.join(folder, f))
	print(C.shape)

	plt.figure()
	sums = np.sum(C, axis=(2, 3))
	pp = np.sum(C, axis=0)[0]
	nn = np.sum(C, axis=0)[1]

	print(sums.shape)
	print(pp.shape)
	print(nn.shape)

	print("sums : %d, pp = %d, nn : %d" % (sums, pp, nn))
	print(np.mean(C, axis=(0, 1)))


	tpr = np.mean(C[:, :, 0, 0]/sums, axis=1) 
	fpr = np.mean(C[:, :, 1, 0]/sums, axis=1) 
	tnr = np.mean(C[:, :, 1, 1]/sums, axis=1) 
	fnr = np.mean(C[:, :, 0, 1]/sums, axis=1) 

	plt.plot(tpr, label="tpr")
	plt.plot(fpr, label="fpr")
	plt.plot(tnr, label="tnr")
	plt.plot(fnr, label="fnr")
	plt.title(f)
	plt.legend(loc="lower right")

plt.show()
