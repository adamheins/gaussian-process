#!/usr/bin/env python3
# coding=utf-8
from functools import partial

import numpy as np
import matplotlib.pyplot as plt


# A small value.
EPSILON = 0.0001


def plot_sigma_bounds(x, m, k, n, color):
    ''' Color the region within n standard deviations of the mean. '''
    upper = m + n * k
    lower = m - n * k
    plt.fill_between(x, lower, upper, color=color)


def covmat(k, X, Y):
    ''' Create a covariance matrix using the kernel function k and vectors of
        random variables X and Y. '''
    return np.array([[k(x, y) for y in Y] for x in X])


def covnew(k, X1, X2):
    ''' Calculate the covariance matrices for new data, accounting for the
        existence of old data. '''
    K12 = covmat(k, X1, X2)
    K21 = K12.T
    K22 = covmat(k, X2, X2)
    return K12, K21, K22


def SEKernel(x, y, sigma=1, lscale=1):
    ''' Squared exponential function kernel. '''
    return sigma**2 * np.exp(-0.5 * (np.linalg.norm(x - y) / lscale)**2)


def RBFKernel(x, y, sigma=1):
    ''' Radial basis function kernel. '''
    return np.exp(-np.linalg.norm(x - y)**2 / (2 * sigma**2))


class GaussianProcess(object):
    def __init__(self, kernel, noise_sigma=0, signal_sigma=1, **kwargs):
        ''' Create a Gaussian Process with given kernel function. '''
        # Additional keyword arguments are passed along to the kernel function.
        self.kernel = partial(kernel, sigma=signal_sigma, **kwargs)
        self.noise_sigma = noise_sigma

        self.X = np.array([])  # Input vector
        self.Y = np.array([])  # Output vector

        self.observed_new_data = False
        self.K11 = np.array([[]])
        self.L = np.array([[]])

    def observe(self, X, Y):
        ''' Take an observation of (X, Y) input-output pairs. '''
        X = np.asarray(X)
        Y = np.asarray(Y)

        if X.shape[0] != Y.shape[0]:
            raise ValueError('First dimension of X and Y must be equal.')

        if self.X.size == 0:
            self.X = X
            self.Y = Y
            self.K11 = covmat(self.kernel, X, X)
        else:
            K12, K21, K22 = covnew(self.kernel, self.X, X)

            n1 = self.X.shape[0]
            n2 = X.shape[0]

            K11 = np.zeros((n1 + n2, n1 + n2))
            K11[:n1, :n1] = self.K11
            K11[:n1, n1:] = K12
            K11[n1:, :n1] = K21
            K11[n1:, n1:] = K22

            self.K11 = K11
            self.X = np.append(self.X, X, axis=0)
            self.Y = np.append(self.Y, Y, axis=0)

        self.observed_new_data = True

    def predict(self, X, outdim=1):
        ''' Predict the output values at the input values contained in X with
            covariance information. '''
        # Output dimensions can be specified to handle the case in which no
        # data has yet been observed.

        X = np.asarray(X)

        if len(self.X) == 0:
            K = covmat(self.kernel, X, X)

            # If we haven't seen any output yet, assume 0-mean.
            mean = np.squeeze(np.zeros((outdim, X.shape[0])))

            return mean, K
        else:
            # If we've seen new data since the last time we calculated K11, we
            # need to recalculate it.
            if self.observed_new_data:
                # Do Cholesky decomposition after adding a small positive value
                # along the diagonal to ensure positive definiteness. Otherwise
                # this goes quite numerically unstable.
                noise = max(EPSILON, self.noise_sigma)
                noise_mat = np.eye(self.K11.shape[0]) * noise
                self.L = np.linalg.cholesky(self.K11 + noise_mat)

                self.observed_new_data = False

            L = self.L
            K12, K21, K22 = covnew(self.kernel, self.X, X)

            a = np.linalg.solve(L.T, np.linalg.solve(L, self.Y))
            v = np.linalg.solve(L, K12)

            # Calculate mean and covariance of the posterior conditional
            # distribution.
            mean = np.squeeze(np.dot(K21, a))
            K = K22 - np.dot(v.T, v)

        return mean, K

    def sample(self, X):
        ''' Sample the GP at input values X and incorporate the results back
            into the GP. This is useful the generating a random function. '''
        mean, cov = self.predict(X)
        Y = np.random.multivariate_normal(mean, cov)
        self.observe(X, Y)

    def plot(self, span=None, step=0.2, sigmas=[0]):
        ''' Plot the GP.'''

        # If span is not passed, it defaults to the range between the minimum
        # and maximum input values.
        if span is None:
            span = (np.min(self.X), np.max(self.X))

        # Predict values over the range of interest. We ensure that all of our
        # actual input values are also predicted to ensure accurate plotting.
        Xi = np.arange(span[0], span[1], step)
        Xi = np.append(Xi, self.X)
        Xi.sort()

        mean, cov = self.predict(Xi)

        # Plot the mean of the learned function.
        _, ax = plt.subplots()
        ax.plot(Xi, mean)

        # Plot sampled points.
        plt.plot(self.X, self.Y, 'x')

        # We explicitly use np.abs(...) because small negative values may
        # appear instead of zeros due to numerical error.
        stddev = np.sqrt(np.abs(np.diag(cov)))
        colors = np.linspace(0.9, 0.7, len(sigmas))

        for sigma, color in zip(sorted(sigmas, reverse=True), colors):
            if sigma > 0:
                plot_sigma_bounds(Xi, mean, stddev, sigma, (color,)*3)

        plt.title('Gaussian Process')

        return plt


def ex1():
    ''' First example: Test a prediction based on some observed data. '''
    gp = GaussianProcess(SEKernel)
    gp.observe([1, 2, 2.4, 3, 4, 5], [1, 2, 2.4, 3, 4, 5])

    sample_point = 2.5
    pred, sdev = gp.predict([sample_point])
    pred = float(pred)
    sdev = float(sdev)

    print(u'Predict {:.4f} at {} with σ = {:.8f}.'.format(pred, sample_point, sdev))

    gp.plot((-3, 8), sigmas=[1, 2, 3])
    plt.plot(np.arange(-3, 8), np.arange(-3, 8))
    plt.show()


def ex2():
    ''' Second example: Plot a function randomly sampled from the GP. '''
    X = np.random.rand(5) * 10
    gp = GaussianProcess(SEKernel)
    gp.sample(X)
    gp.plot((0, 10), sigmas=[1, 2, 3]).show()


def main():
    ex1()
    ex2()


if __name__ == '__main__':
    main()
