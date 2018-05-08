import numpy as np
import scipy.io as sio
from os import path
import feature_utils as utils

# Select the features to calculate 
# Either "hog_9" or "histograms"
f_name = "hog_9"

images = utils.images()
for i, image_cell in enumerate(images):
	image_path = path.join(utils.folder_images, image_cell[0][0])
	print("Image : %5i, %s" % (i, image_path))
	feature_path = path.join(utils.folder_features, f_name, "%0fd.mat" % i)
	if f_name == "histograms":
		utils.histogramFeatures(image_path, n=1, path=feature_path)
	elif f_name == "hog_9":
		utils.hogFeatures(image_path, n=9, path=feature_path)