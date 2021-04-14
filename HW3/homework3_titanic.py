import pandas
import numpy as np
from homework3_twgoon import *

if __name__ == "__main__":
    # Load training data
    d = pandas.read_csv("data\\titanic\\train.csv")
    y = d.Survived.to_numpy()
    sex = d.Sex.map({"male":0, "female":1}).to_numpy()
    Pclass = d.Pclass.to_numpy()
    sib = d.SibSp.to_numpy()

    # convert Pclass to catigorical data
    Pclass = one_hot(Pclass-1, 3)

    sex = np.array([sex])
    sib = np.array([sib])

    x_train = np.concatenate((sex.T, Pclass), axis=1)
    x_train = np.concatenate((x_train, sib.T), axis=1)

    # Train model using part of homework 3.
    W = np.random.randn(5, 2)
    W = softmaxRegression(W, x_train.T, y, None, None, 2, epsilon=0.1, batchSize=99, epoches=200)

    # Load testing data
    d_test = pandas.read_csv("data\\titanic\\test.csv")
    id_test = d_test.PassengerId.to_numpy()
    sex_test = d_test.Sex.map({"male":0, "female":1}).to_numpy()
    Pclass_test = d_test.Pclass.to_numpy()
    sib_test = d_test.SibSp.to_numpy()

    # convert Pclass to catigorical data
    Pclass_test = one_hot(Pclass_test-1, 3)

    sex_test = np.array([sex_test])
    sib_test = np.array([sib_test])

    x_test = np.concatenate((sex_test.T, Pclass_test), axis=1)
    x_test = np.concatenate((x_test, sib_test.T), axis=1)

    # Compute predictions on test set
    p = predict(W, x_test.T)
    print("Training Loss: ", CE_loss(W, x_train.T, y, 2))
    print("Training Accuracy: ", calc_accuracy(W, x_train.T, y))

    # Write CSV file of the format:
    # PassengerId, Survived
    id_test = np.array([id_test])
    p = np.array([p])
    csv = np.concatenate((id_test.T, p.T), axis=1)

    file = pandas.DataFrame(csv, columns=["PassengerId", "Survived"])
    file.to_csv("twgoon_predictions.csv", index=False)
