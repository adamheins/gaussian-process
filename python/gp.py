import numpy as np
import matplotlib.pyplot as plt


def covmat(k, X, Y):
    ''' Create a covariance matrix using the kernel function k and vectors of
        random variables X and Y. '''
    return np.array([[k(x, y) for y in Y] for x in X])


def SEKernel(x, y, stddev=1, lscale=1):
    ''' Squared exponential function kernel. '''
    return stddev**2 * np.exp(-0.5 * (np.linalg.norm(x - y) / lscale)**2)


def RBFKernel(x, y, stddev=1):
    ''' Radial basis function kernel. '''
    return np.exp(-np.linalg.norm(x - y)**2 / (2 * stddev**2))


class GaussianProcess(object):
    def __init__(self, kernel):
        ''' Create a Gaussian Process with given kernel function. '''
        self.kernel = kernel
        self.X = np.array([])  # Input vector
        self.Y = np.array([])  # Output vector

    def observe(self, X, Y):
        ''' Take an observation of (X, Y) input-output pairs. '''
        X = np.asarray(X)
        Y = np.asarray(Y)

        if X.shape[0] != Y.shape[0]:
            raise ValueError('First dimension of X and Y must be equal.')

        if self.X.size == 0:
            self.X = X
            self.Y = Y
        else:
            self.X = np.append(self.X, X, axis=0)
            self.Y = np.append(self.Y, Y, axis=0)

    def predict(self, X, outdim=1):
        ''' Predict the output values at the input values contained in X with
            covariance information. '''
        # Output dimensions can be specified to handle the case in which no
        # data has yet been observed.

        X = np.asarray(X)

        if len(self.X) == 0:
            K = covmat(self.kernel, X, X)
            return np.squeeze(np.zeros((outdim, X.shape[0]))), K
        else:
            K11 = covmat(self.kernel, self.X, self.X)
            K12 = covmat(self.kernel, self.X, X)
            K21 = covmat(self.kernel, X, self.X)
            K22 = covmat(self.kernel, X, X)

            # Do Cholesky decomposition after adding a small positive value
            # along the diagonal to ensure positive definiteness. Otherwise
            # this goes quite numerically unstable.
            L = np.linalg.cholesky(K11 + np.eye(K11.shape[0]) * 0.0001)

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

    # Test a prediction based on some observed data.
    gp1 = GaussianProcess(SEKernel)
    gp1.observe([1, 2, 3, 4, 5], [1, 2, 3, 4, 5])
    pred, sdev = gp1.predict([2.5])
    print('Predict {} at {} with std. dev. of {}.'.format(float(pred), 2.5, sdev))

    # Input/index points.
    # X = np.arange(sample_span[0], sample_span[1], sample_step)
    X = np.random.rand(5) * 10

    # Plot the function based a function randomly sampled from the GP.
    gp2 = GaussianProcess(SEKernel)
    gp2.sample(X)
    gp2.plot(sample_span, sigma=2)


if __name__ == '__main__':
    main()
