# Machine Learning techniques for time series classification

This repository shows different approaches to time series classification using machine learning techniques. In addition to comparing some of the most used machine learning models (Deep learning, Neural network, Support Vector Machine ...), we oppose two different types of input for some of these models: using the raw time series or using features extracted from the time series (statistical measures, frequency domain features, geometrical features... features detailed below). 

### Time series data: Human Activity Recognition (HAR data)

The data set we use in this repository is a Human Activity Recognition database available for free [here](https://archive.ics.uci.edu/ml/datasets/Human+Activity+Recognition+Using+Smartphones) on UCI Machine Learning Repository.
This data set consists in 3-axial accelerometer and gyroscope signals captured with a smartphone worn on the waist while performing 6 different activities:  

 - Walking  
 - Walking upstairs  
 - Walking downstairs  
 - Sitting  
 - Standing  
 - Laying  

Each sample in the dataset is a 2.56s window sampled at a 50Hz rate which makes 6 x 128 readings per sample (3 accelerometer axis x,y,z and 3 gyroscope axis x,y,z). In this repository we will only use the 3 accelerometer axis x, y and z. 

Below we plot and example of the time series for each activity:

![Accelerometer walking](https://octodex.github.com/images/plots_walking.png)
![Accelerometer walking upstairs](https://octodex.github.com/images/plots_upstairs.png)
![Accelerometer walking downstairs](https://octodex.github.com/images/plots_downstairs.png)
![Accelerometer sitting](https://octodex.github.com/images/plots_sitting.png)
![Accelerometer standing](https://octodex.github.com/images/plots_standing.png)
![Accelerometer laying](https://octodex.github.com/images/plots_laying.png)

The classification task here consists in recognizing the 6 activities given above. As explained in the introduction, we will either use the raw signals (plots above) or features extracted from these signals. 

### Features extraction

In some of the notebooks in this repository we will extract features from the time series and use them as input for the Machine Learning models. In these notebooks we will build helper functions to extract the desired features. 
We extract  statistical and geometrical features from raw signals and jerk signals (acceleration first derivative), frequency domain features from raw signals and jerk signals. The full list of features is:

 - **x,y and z raw signals** : mean, max, min, standard deviation, skewness, kurtosis, interquartile range, median absolute deviation, area under curve, area under squared curve
 - **x,y and z jerk signals (first derivative)** : mean, max, min, standard deviation, skewness, kurtosis, interquartile range, median absolute deviation, area under curve, area under squared curve
 - **x,y and z raw signals Discrete Fourrier Transform**: mean, max, min, standard deviation, skewness, kurtosis, interquartile range, median absolute deviation, area under curve, area under squared curve, weighted mean frequency, 5 first DFT coefficients, 5 first local maxima of DFT coefficients and their corresponding frequencies.
 - **x,y and z jerk signals Discrete Fourrier Transform**: mean, max, min, standard deviation, skewness, kurtosis, interquartile range, median absolute deviation, area under curve, area under squared curve, weighted mean frequency, 5 first DFT coefficients, 5 first local maxima of DFT coefficients and their corresponding frequencies.
 - **x,y and z correlation coefficients**
 

### Machine Learning Approaches in this repository

The 6 different approaches in this repository are:  

- **Convolutional Neural Network with raw data**: a deep learning method largely used for computer vision (2D inputs) but in our case we adapt it to time series data (1D inputs). So instead of 2D convolution layers, we use 1D convolution layers by setting the convolutional height to 1. We add L2 regularization and dropout to prevent overfitting (we only use L2 regularization in this case).

- **k Nearest Neighbors with raw data**: in this repository we implement a brute force kNN algorithm and use two different similarity measures for the time series: the euclidian distance and the Dynamic Time Warping distance (DWT). DWT overcomes the problem of synchronisation between two similar time series by associating points that might not face each other. 

- **k Neareast Neighbors (kNN) with features**: we use the same algorithm presented above but we use features extracted from the time series (features detailed below). In this case we only use euclidian distance to measure similiarity between samples as we are now considering n dimensional vector samples (n being the number of features extracted) and no longer time series.

- **Neural Network with raw data**: we build a 2 layer neural network with softmax activation and L2 regularization. We feed this network with raw data. 

- **Neural Network with features**: we build a 2 layer neural network with softmax activation and L2 regularization and feed it with extracted features from raw data.

- **Support Vector Machine (SVM) with features**: We feed a SVM with RDF kernel with features extracted from raw data. We optimize C and gamma hyperparameters to achieve the best f1 score on the test set.



