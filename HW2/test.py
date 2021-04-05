import numpy as np

a = np.arange(9)
b = np.reshape(a, (3,3))
print(b)
print(np.reshape(b, (9)))