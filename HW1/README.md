# HW 1 Part 1

My solutions to a variety of different matrix operation problems to do in numpy and an associated test harness.

More details on this assignment can be found in "homework1.pdf".

## Problems
1. Given matricies A and B, compute A+B
2. Given matricies A, B, and C, compute AB-C
3. Given matricies A, B, and C, compute A⊙B+transpose(C)
4. Given column vectors x and y and square matrix S, compute transpose(x)Sy
5. Given matrix A, return a matrix with the same dimensions as A filled with all zeros
6. Given matrix A, return a matrix with the same dimensions as A filled with all ones
7. Given square matrix A and scalar _a_, compute A+_a_I, where I is the identity matrix with the same dimensions as A
8. Gevn matrix A and integers _i_, _j_, return the _j_th column of the _i_th row of A
9. Given matrix A and integer _i_, return the sum of all entries in the _i_th row
10. Given matrix A and scalars _c_, _d_, compute the arithmetic mean over all entries of A that are between _c_ and _d_ inclusive
11. Given n x n matrix A, and integer _k_, return an n x k matrix containing the right-eigenvectors of A corresponding to the _k_ largest eigenvalues of A
12. Given square matrix A and column vector x, use `np.linalg.solve` to compute A^(-1)x
13. Given square matrix A and row vector x, use `np.linalg.solve` to compute xA^(-1)

## Files

Code for this part is in "homework1_twgoon.py".

Tests are in "tests.py".

## Usage
To run: `python homework1_twgoon.py`

# HW1 1 Part 2

A ensemble classifier to detect smiles in 24 x 24 greyscale images. 5 classifiers in total was used and each classifier in the ensemble compares the values of 2 pixels in the image. The model was trained by using stepwise regression where the pair of pixels in a grid search that performed the best were added to the ensemble. A visualization of the chosen pixels are shown below.

![image](https://user-images.githubusercontent.com/32044950/120902442-79c92800-c60e-11eb-94bd-f68047eb38fb.png)

More details on the assignment can be found in "homework1.pdf".

## Files

Code for this part is in "homework1_smile_twgoon.py".

## Usage
For "homework1_smile_twgoon.py" to work correctly the file structure must be maintained in the following format:
homework1_smile_twgoon.py

    data/
        testing/
            testingFaces.npy
            testingLabels.npy
        training/
            trainingFaces.npy
            trainingLabels.npy
            
To run: `python homework1_smile_twgoon.py`
