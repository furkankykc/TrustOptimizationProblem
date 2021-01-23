import math
import numpy as np
from scipy.optimize import minimize
from scipy.optimize import Bounds
import matplotlib.pyplot as plt

# fitness = lambda x: math.pow(x[0], 0.6) + math.pow(x[1], 0.6) - 6 * x[0] - 4 * x[2] + 3 * x[3]


inputs = []
results = []


# fitness function for given equation x^0.6 + y^0.6 - 6*x - 4z + 3k
def fitness(x):
    global inputs
    inputs.append(x)
    res = x[0] ** 0.6 + x[1] ** 0.6 + -6 * x[0] - 4 * x[2] + 3 * x[3]
    results.append(res)
    return res


# Boundary equation system
A = np.array([[0, 1, 0, 2], [0, 0, 0, 1], [1, 0, 2, 0], [1, 0, 0, 0]])
# Boundary Values
b = np.array([4, 1, 4, 3])

# Minimum values for boundaries
x_min = [0, 0, 0, 0]
# Solution of the equation for given upper boundaries
x_max = np.linalg.solve(A, b)
print(x_max)

# Initial x values
x0 = np.array([1, 1, 0.4, 0.8])
# Definition of bounds
bounds = Bounds(x_min, x_max)
# minimization with trust-constr algorithm for given bounds,initial x and the fitness value
res = minimize(fitness, x0, method='trust-constr', bounds=bounds, jac='2-point')

print(res.x)
print('min=', res.fun)

plt.plot(inputs)
plt.legend(["x1", "x2", "x3", "x4"])
plt.show()

plt.plot(results)
plt.show()
