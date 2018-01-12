import scipy.integrate as integrate
import numpy as np


A = np.eye(2)
B = np.zeros((2, 2))
C = np.zeros((2, 2))


def u(t):
    ''' Control signal as a function of time. '''
    return np.array([0, 0])


def f(x, t):
    ''' State-update function. '''
    return np.dot(A, x) + np.dot(B, u(t))


def g(x):
    ''' Output function. '''
    return np.dot(C, x)


t = np.arange(0, 2, 0.1)
x0 = np.array([0, 0])
x = integrate.odeint(f, x0, t, args=(u,))
print(x.shape)
