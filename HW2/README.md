# HW 2 Linear Regression for Age Estimation

A linear regression model trained in 3 ways to predict the ages of people from 48 x 48 greyscale images of their face.

3 methods:
- Solving for the optimal linear regression weights with ![image](https://user-images.githubusercontent.com/32044950/120902901-e7765380-c610-11eb-9d08-620cae98b9ba.png)
- Finding the optimal weights with gradient descent (without regularization)
- Finding the optimal weights with gradient descent (with regularization)

Also includes a variable degree polynomial regression model.

More details on the assignment can be found in "homework2.pdf".

Some performance data on the trained linear regression models can be found in "homework2_errors_twgoon.pdf".

## Usage
For "homework2_twgoon.py" to work correctly the file structure must be maintained in the following format:

    homework2_twgoon.py
    data/
        testing/
            age_regression_Xte.npy
            age_regression_yte.npy
        training/
            age_regression_Xtr.npy
            age_regression_ytr.npy
            
To run: `python homework2_twgoon.py`
