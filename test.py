import numpy as np
import rbf

#x = np.array([[1, 2, 3], [4, 5, 6]])
#a = x.shape
#b = x.shape[1]
x = np.arange(0, 5, 1)
degree = 4
N = x.shape[0]
X = np.zeros((N, degree + 1))

for j in range(0, degree+1): #loop from 0 to degree+1 -1 pay attention
    powerj = np.power(x, j)
    X[:, j] = powerj

print(X)

centers, sigma = rbf.get_centers_and_sigma(2)
design
