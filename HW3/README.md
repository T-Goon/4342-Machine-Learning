# HW 3 Softmax Regression and Kaggle

## Fasion Mnist
A softmax (aka multinomial logistic regression) model was trained to predict clothing types from 28 x 28 greyscale images from the Fashion Mnsit dataset. The model was trained using stochastic gradient descent with a cross entropy loss function.

Code for this can be found in "homework3_twgoon.py".

More details on the assignment can be found in "homework3.pdf".

### Results and Visualizations

![image](https://user-images.githubusercontent.com/32044950/120903218-ec3c0700-c612-11eb-864e-def0d58a1b7d.png)

![image](https://user-images.githubusercontent.com/32044950/120903226-f52cd880-c612-11eb-901a-15ff4b80efea.png)

These and a bit more can be found in "homework3_twgoon.pdf".

### Usage

The file structure should be:

    homework3_twgoon.py
    data/
      training/
        fashion_mnist_train_images.npy
        fashion_mnist_train_labels.npy
      testing/
        fashion_mnist_test_images.npy
        fashin_mnist_test_labels.npy
        
To run: `python homework3_twgoon.py`

## Titanic
The softmax regression model was also applied to the Titanic competition on Kaggle. https://www.kaggle.com/c/titanic.

### Results

77.3% accuracy on the Kaggle test set.

A screenshot can be found in "homework3_twgoon.pdf".

More details on the assignment can be found in "homework3.pdf".

### Usage
The file structure should be:
    
    homework3_twgoon.py
    homework3_titanic.py
    data/
      titanic/
        train.csv

To run: `python homework3_titanic.py`
