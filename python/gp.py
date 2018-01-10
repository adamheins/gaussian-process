from functools import partial

import numpy as np
import matplotlib.pyplot as plt
from scipy import interpolate as interp


def covmat(k, X, Y):
    ''' Create a covariance matrix using the kernel function k and vectors of
        random variables X and Y. '''
    return np.matrix([[k(x, y) for x in X] for y in Y])


def se(x, y, stddev=1, lscale=1):
    ''' Squared exponential function kernel. '''
    return stddev**2 * np.exp(-0.5 * ((x - y) / lscale)**2)


def rbf(x, y, stddev=1):
    ''' Radial basis function kernel. '''
    return np.exp(-(x - y)**2 / (2 * stddev**2))


def plot(X, F, interp_step_size):
    ''' Plot sampled points and iterpolated spline. '''
    # Fit a spline to the data.
    query_step = (X[1] - X[0]) * interp_step_size
    query_pts = np.arange(X[0], X[-1] + query_step, query_step)
    spline = interp.InterpolatedUnivariateSpline(X, F)

    # Plot the spline.
    plt.plot(query_pts, spline(query_pts))

    # Plot sampled points.
    plt.plot(X, F, 'x')

    plt.title('Random Sampling of Gaussian Process')
    plt.show()


def main():
    lscale = 1 # Length scale.
    stddev = 1 # Standard deviation.
    num_samples = 100 # Number of random samples.
    sample_step = 0.5 # Step size of index variable.

    # Input/index points.
    X = np.arange(0, num_samples * sample_step, sample_step)

    # Zero mean.
    mean = np.zeros(num_samples)

    # Generate covariance matrix.
    k = partial(se, stddev=stddev, lscale=lscale)
    K = covmat(k, X, X)

    # Get a random sample of outputs based on the inputs.
    F = np.random.multivariate_normal(mean, K)

    plot(X, F, 0.2)


if __name__ == '__main__':
    main()
