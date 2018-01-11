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
        ''' Create a Gaussian Process with given kernel function. '''
        self.kernel = kernel
        self.X = np.array([])  # Input vector
        self.Y = np.array([])  # Output vector

    def observe(self, X, Y):
        ''' Take an observation of (X, Y) input-output pairs. '''
        self.X = np.append(self.X, np.asarray(X))
        self.Y = np.append(self.Y, np.asarray(Y))

    def predict(self, X):
        ''' Predict the output values at the input values contained in X with
            covariance information. '''
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
        ''' Sample the GP at input values X and incorporate the results back
            into the GP. This is useful the generating a random function. '''
        mean, cov = self.predict(X)
        Y = np.random.multivariate_normal(mean, cov)
        self.observe(X, Y)

    def plot(self, span=None, step=0.2, sigma=0):
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

        if sigma > 0:
            # Plot uncertainty bounds of the learned function. We explicitly
            # use np.abs(...) because small negative values may appear instead
            # of zeros due to numerical error.
            stddev = np.sqrt(np.abs(np.diag(cov)))
            upper = mean + 2 * stddev
            lower = mean - 2 * stddev
            ax.fill_between(Xi, lower, upper, color=(0.8, 0.8, 0.8))

        plt.title('Gaussian Process')
        plt.show()


def main():
    sample_span = (0, 10)
    sample_step = 1

    # Test a predict based on some observed data.
    gp1 = GP(se)
    gp1.observe([1, 2, 3, 4, 5], [1, 2, 3, 4, 5])
    pred, sdev = gp1.predict([2.5])
    print('Predict {} at {} with std. dev. of {}.'.format(float(pred), 2.5, sdev))

    # Input/index points.
    # X = np.arange(sample_span[0], sample_span[1], sample_step)
    X = np.random.rand(5) * 10

    # Plot the function based a function randomly sampled from the GP.
    gp2 = GP(se)
    gp2.sample(X)
    gp2.plot(sample_span)


if __name__ == '__main__':
    main()
