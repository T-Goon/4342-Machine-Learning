# HW 6 Neural Networks

A 3 layer neural network implemented from scratch using numpy to predict the clothing types from 28 x 28 greyscale images from the Fashion Mnist dataset. Optimization was done using gradient descent with the cross entropy loss function and both L1 and L2 regularization. Also includes code to do hyperparameter optimization.

![image](https://user-images.githubusercontent.com/32044950/120904003-9b7add00-c617-11eb-8cb7-bb8c26b8994d.png)


More details on the assignment can be found in "homework6.pdf".

## Results

About 86% accuracy.

## Usage
To run the program maintain the file structure of :

    homework6_twgoon.py
	    data/
		    test/
	    		fashion_mnest_test_images.npy
	    		fashion_mnest_test_labels.npy
    		train/
		    	fashion_mnest_train_images.npy
	    		fashion_mnest_train_labels.npy
