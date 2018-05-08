import numpy as np
import scipy.io as sio
from skimage import color, exposure
from skimage.io import imread
import matplotlib.pyplot as plt

image_path1 = "../data/images/s/stone_circle/sun_anxzechmggrinczx.jpg"
image_path2 = "../data/images/s/stone_circle/sun_agdbqxlmmdjvovpx.jpg"

image1 = imread(image_path1)
image2 = imread(image_path2)

#image1 = color.rgb2gray(imread(image_path1))
#image2 = color.rgb2gray(imread(image_path2))

if len(image1.shape) > 3: image1 = image1[0]

print(len(image1.shape))
print(len(image2.shape))
print(image1.shape)
print(image2.shape)

plt.imshow(image1)
plt.show()