from functools import partial

import numpy as np
import matplotlib.pyplot as plt
from scipy import interpolate as interp


def spline(X, Y, interp_step_size):
    query_step = (X[1] - X[0]) * interp_step_size
    query_pts = np.arange(X[0], X[-1] + query_step, query_step)
    spline = interp.InterpolatedUnivariateSpline(X, Y)
    return query_pts, spline(query_pts)


def covmat(k, X, Y):
    ''' Create a covariance matrix using the kernel function k and vectors of
        random variables X and Y. '''
    return np.matrix([[k(x, y) for y in Y] for x in X])


def se(x, y, stddev=1, lscale=1):
    ''' Squared exponential function kernel. '''
    return stddev**2 * np.exp(-0.5 * ((x - y) / lscale)**2)


def rbf(x, y, stddev=1):
    ''' Radial basis function kernel. '''
    return np.exp(-(x - y)**2 / (2 * stddev**2))


class GP(object):
    def __init__(self, kernel):
        self.kernel = kernel
        self.X = np.array([]) # Input vector
        self.Y = np.array([]) # Output vector


    def observe(self, X, Y):
        self.X = np.append(self.X, np.asarray(X))
        self.Y = np.append(self.Y, np.asarray(Y))


    def predict(self, X):
        X = np.asarray(X)
        if len(self.X) == 0:
            K = covmat(self.kernel, X, X)
            mean = np.zeros_like(X)
        else:
            K11 = covmat(self.kernel, self.X, self.X)
            K12 = covmat(self.kernel, self.X, X)
            K21 = covmat(self.kernel, X, self.X)
            K22 = covmat(self.kernel, X, X)

            K11_inv = np.linalg.inv(K11)

            # Calculate mean and covariance of the posterior conditional
            # distribution.
            K = K22 - K21 * K11_inv * K12
            mean = K21 * K11_inv * np.transpose(np.asmatrix(self.Y))

            # Convert from matrix to array.
            mean = mean.A1

        return mean, K


    def sample(self, X):
        mean, cov = self.predict(X)
        Y = np.random.multivariate_normal(mean, cov)
        self.observe(X, Y)


    def plot(self, interp_step_size=0.2):
        Xi, Yi = spline(self.X, self.Y, interp_step_size)

        # Plot the spline.
        plt.plot(Xi, Yi)

        # Plot sampled points.
        plt.plot(self.X, self.Y, 'x')

        _, cov = self.predict(Xi)
        dev = np.sqrt(np.diag(cov))
        print(dev)
        # foo, upper = spline(Xi, Yi + 2 * dev, interp_step_size)
        # _, lower = spline(Xi, Yi - 2 * dev, interp_step_size)
        # plt.plot(foo, upper)
        # plt.plot(foo, lower)

        plt.show()


def main():
    num_samples = 100 # Number of random samples.
    sample_step = 0.5 # Step size of index variable.

    # Input/index points.
    X = np.arange(0, num_samples * sample_step, sample_step)

    gp1 = GP(se)
    gp1.observe([1, 2, 3, 4, 5], [1, 2, 3, 4, 5])
    pred, sdev = gp1.predict([2.5])
    print('Predict {} at {} with std. dev. of {}.'.format(float(pred), 2.5, sdev))

    gp2 = GP(se)
    gp2.sample(X)
    gp2.plot()


if __name__ == '__main__':
    main()
