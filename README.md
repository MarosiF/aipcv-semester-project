# Advanced Image Processing and Computer Vision (AIPCV) Semester Project 

This project is about using image processing techniques to solve a scene recognition problem. The database used for this project is the [Sun Attribute Database](https://cs.brown.edu/~gen/sunattributes.html). The feature vecotrs used are the GIST descriptor from _Oliva, Aude, and Antonio Torralba. "Modeling the shape of the scene: A holistic representation of the spatial envelope." International journal of computer vision 42.3 (2001): 145-175._, the Histogram of Oriented Gradients as implemented in [scikit-image](http://scikit-image.org/docs/dev/api/skimage.feature.html#skimage.feature.hog) and a color histogram for comparison.

# Requirements
The code for the GIST descriptor needs to be downloaded from [Modeling the shape of the scene: a holistic representation of the spatial envelope](http://people.csail.mit.edu/torralba/code/spatialenvelope/) and extracted into the __MATLAB__ folder. The Sun Attribute Database is also necessary to be downloaded seperatelly to run this project.

## Required Python Libraries
- [numpy](http://www.numpy.org/)
- [scipy](http://www.scipy.org/)
- [scikit-image](http://scikit-image.org/)
- [scikit-learn](http://scikit-learn.org/)
- [matplotlib](http://matplotlib.org/)

# Usage
In order to run this project the attributes and images have to be downloaded and extracted. Further, the filepaths in __features_utils.py__ have to be updated accordingly to the downloaded files in the __Sun Attribute Database__. To extract features use either [calculate_gist_features.m](MATLAB/calculate_gist_features.m) (again change filepaths accordingly) for the GIST features or use [calculate_features.py](calculation_features.py) (select feature in the file). The features are then transformed from mat files to npy files to be loaded faster by using the [parse_features.py](parse_features.py) file (addapt the parameters in the accordingly). Finaly use any of the files with the __testing__ prefixes to run the calculations for the classifications. The __results__ folder is where all the results are stored and in the __MATLAB__ folder is everything needed for the GIST feature descriptor calculation.

# Contributors
Nikolai Janakiev, [Thomas Renoth](https://github.com/minus7), Juan Fernandez Troyano

# License
This project is licensed under the MIT license. See the [LICENSE](LICENSE) for details.
