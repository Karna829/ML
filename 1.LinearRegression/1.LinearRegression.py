import pandas as pd
import numpy as np


class LinearRegression():
    def __init__(self, learning_rate=0.01, iterations=1000, normalize=False, verbose=False):
        self.icost = 0  # initial cost of regression
        self.lrate = learning_rate
        self.iterations = iterations
        self.normalize = normalize
        # used to store mean of data during normalization
        self.dmean = np.array([])
        # used to store variance of data during normalization
        self.dvar = np.array([])
        self.has_bias = True    # Assumed that data has bias term
        self.verbose = verbose

    def fit(self, X, Y):
        x = np.array([])
        y = Y
        self.theta = np.random.rand(1, X[0].shape[0])
        if self.normalize:
            self.dmean = np.mean(X[:, 1:], axis=0)
            self.dvar = np.var(X[:, 1:], axis=0)
            x = np.ones_like(X)
            x[:, 1:] = (X[:, 1:] - self.dmean)/self.dvar
        else:
            x = X
            y = Y

        sample_size = x.shape[0]
        for iteration in range(self.iterations):
            grad_theta = np.zeros_like(self.theta)
            cost = 0
            for sample in range(sample_size):
                scalar = np.dot(self.theta, np.array(x[sample])) - y[sample]
                cost += scalar**2
                grad_theta += scalar*x[sample]
            cost /= (2*sample_size)
            grad_theta /= sample_size
            if iteration == 0:
                self.icost = cost
            self.theta += (-1*self.lrate*grad_theta)
            if self.verbose:
                print('epoch [{}] cost {} '.format(iteration, cost))

    def predict(self, X):
        if self.normalize:
            x = np.ones_like(X)
            for i in range(1, X.shape[0]):
                x[i] = (X[i] - self.dmean[i-1])/self.dvar[i-1]
        else:
            x = X
        predicted = np.dot(self.theta, x)
        return predicted


def main():
    ex1data1 = r'res/ex1data1.txt'
    ex1data2 = r'res/ex1data2.txt'

    data1 = np.genfromtxt(ex1data1, delimiter=',')
    data2 = np.genfromtxt(ex1data2, delimiter=',')
    # Add intercept term
    X1 = np.ones(data1.shape, dtype=float)
    X1[:, 1:] = data1[:, 0:-1]
    Y1 = data1[:, -1]
    # Add intercept term
    X2 = np.ones(data2.shape, dtype=float)
    X2[:, 1:] = data2[:, 0:-1]
    Y2 = data2[:, -1]
    # Linear Regression with single feature [x1 x2] with x1=1
    regressor1 = LinearRegression(
        iterations=1500, learning_rate=0.01)
    regressor1.fit(X1, Y1)
    value = regressor1.predict(np.array([1, 5.5277]))
    print('Expected value is {}  predicted is {} '.format(9.1302, value))
    # Linear Regression with multi feature [x1 x2 x3] with x1=1
    regressor2 = LinearRegression(
        iterations=800, learning_rate=0.01, normalize=True)
    regressor2.fit(X2, Y2)
    value = regressor2.predict(np.array([1, 2104, 3]))
    print('Expected value is {}  predicted is {} '.format(399900, value))


if __name__ == '__main__':
    main()
