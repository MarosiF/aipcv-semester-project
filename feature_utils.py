import numpy as np
import scipy.io as sio
from skimage import color, exposure
from skimage.io import imread
from skimage.feature import hog
from skimage.transform import resize
from os import path


# Change these folders accordingly
path_labels = "../data/sun_attributes/attributeLabels_continuous.mat"
path_attributes = "../data/sun_attributes/attributes.mat"
path_images = "../data/sun_attributes/images.mat"
folder_features = "../features"
folder_images = "../data/sun_images"


# Loading the mat files with the meta data
def labels(): 		return sio.loadmat(path_labels)['labels_cv']
def attributes(): 	return sio.loadmat(path_attributes)['attributes']
def images(): 		return sio.loadmat(path_images)['images']


# name ... Feature to load (histograms, histograms_2x2, histogram_4x4, gist, hog_9, hog_18)
def load_features(name):
	if name == 'histograms':
		X = np.load(path.join(folder_features, "histograms.npy"))
		y = labels() > 0.5

	elif (name == 'histograms_2x2') | (name == 'histograms_4x4'):
		X0 = np.load(path.join(folder_features, name+".npy"))
		y0 = labels() > 0.5		
		n_samples, n_features, n = X0.shape
		n_targets = y0.shape[1]

		X = np.ndarray((n_samples*n, n_features))
		y = np.ndarray((n_samples*n, n_targets), dtype=bool)

		for i, idx in enumerate(range(0, n*n_samples, n)):
			X[idx:idx+n] = X0[i].T
			y[idx:idx+n] = np.ones((n, n_targets))*y0[i]
	
	elif name == 'gist':
		X = np.load(path.join(folder_features, "gist.npy"))
		y = labels() > 0.5

	elif (name == 'hog_9') | (name == 'hog_18'):
		X = np.load(path.join(folder_features, '%s.npy' % name))
		y = labels() > 0.5

	else: raise Exception('features \"%s\" not found' % name)

	return X, y


# image path 	... path of image
# n 			... number of patches bins
# ṕath 			... output path
def histogramFeatures(image_path, n=1, path=None):
	image = imread(image_path)
	# Take first frame if picture is animated (image index : 12015)
	if len(image.shape) > 3: image = image[0]
	image = color.rgb2gray(image).astype(float)

	histList = []
	if n > 1:
		v_splits = np.array_split(image, n, axis=0)
		for v_image in v_splits:
			h_splits = np.array_split(v_image, n, axis=1)
			for block in h_splits: 
				histogram = exposure.histogram(block, nbins=256)
				h = histogram[0].astype(np.float64)
				h = h/np.sum(h)
				histList.append(h)
	elif n == 1:
		histogram = exposure.histogram(image, nbins=256)
		h = histogram[0].astype(np.float64)
		h = h/np.sum(h)
		histList.append(h)

	histList = np.array(histList)
	if path != None: 
		sio.savemat(path, {'hists' : histList})
	else:
		return histList

# image path 	... path of image
# n 			... number of histogram bins
# ṕath 			... output path 
def hogFeatures(image_path, n=9, path=None):
	image = imread(image_path)
	# Take first frame if picture is animated (image index : 12015)
	if len(image.shape) > 3: image = image[0]
	image = color.rgb2gray(image)

	# Resize image
	image = resize(image, (256, 256))
	fd = hog(image, orientations=9, pixels_per_cell=(32, 32), cells_per_block=(1, 1))
	fd = np.array([fd])

	if path != None: 
		sio.savemat(path, {'hog' : fd})
	else:
		return fd

