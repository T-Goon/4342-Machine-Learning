import numpy as np
import matplotlib.pyplot as plt
def reshapeAndAppend1s (faces):

    # Flatten images
    faces = np.reshape(faces, (-1, faces.shape[1]**2))

    # Append one onto each image
    ones = np.ones((faces.shape[0], 1))
    faces = np.concatenate((faces, ones), axis=1)

    faces = faces.T
    # print("append ",faces.shape)

    return faces
# Load data
Xtilde_te = reshapeAndAppend1s(np.load("data\\testing\\age_regression_Xte.npy"))
yte = np.load("data\\testing\\age_regression_yte.npy")

with open("w.npy", "rb") as f:
    w1 = np.load(f)
    w2 = np.load(f)
    w3 = np.load(f)

# predictions = np.dot(Xtilde_te.T, w3)
# print(np.sqrt(np.mean((predictions-yte)**2)))
# print(np.sort((predictions-yte)**2))
# errors = (predictions-yte)**2
# index = list()
# for i in range(5):
#     max = np.argmin(errors)
#     index.append(max)
#     errors[max] = float('inf')

# print(index)
# print(((predictions-yte)**2)[index])
# print(predictions[index])
# print(yte[index])

# print(Xtilde_te.shape)
# for i in index:
#     print(i)
#     im = Xtilde_te.T[i, :Xtilde_te.shape[0]-1]
#     im = np.reshape(im, (48, 48))
#     plt.imshow(im) 
#     plt.show()
# print(w1[-1:])

# w1 = w1[:-1]
# im = np.reshape(w1.T, (48, 48))
# plt.imshow(im) 
# plt.title("Anlytical Method")
# plt.show()

print(w2[:-1])
print(w2[:-1].reshape((48, 48)))

plt.imshow(w2[:-1].reshape((48, 48))) 
plt.title("Gradient Descent Method")
plt.show()

plt.imshow(w3.T[:-1].reshape((48, 48))) 
plt.title("Anlytical Method")
plt.show()

# def trainPolynomialRegressor (x, y, d):
#     samples = np.ones((x.shape[0], 1))
#     print(samples)
    
#     # create feature that are powers of x
#     for i in range(1, d+1):
#         x_pow = x**i
#         x_pow = np.reshape(x_pow, (-1, 1))
#         print(x_pow.shape)
#         samples = np.concatenate((samples, x_pow), axis=1)
    
#     w = np.linalg.solve(
#         np.dot(samples.T, samples),
#         np.dot(samples.T, y)
#     )

#     print(samples)

#     print(w)
#     return w

# w = trainPolynomialRegressor(np.arange(10), np.arange(10)**3, 3)

# print(np.dot(np.array([1.,   4.,  16. , 64.]), w))