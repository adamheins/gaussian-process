#!/usr/bin/env python3
# coding=utf-8
from __future__ import print_function

import time
import sys

import scipy.integrate as integrate
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D

from gp import GaussianProcess, SEKernel, plot_sigma_bounds


# Physical pendulum parameters.
GRAV = 9.81
LENGTH = 1
MASS = 1
INERTIA = MASS * LENGTH ** 2

# Times are in seconds.
DT = 0.05
T0 = 0
TF = 8

# Initial condition.
X0 = np.array([np.pi * 0.75, 0])

# Assumed standard deviation of signal for Gaussian process regression.
SIGNAL_SIGMA = 1.0

NOISE_MEAN = 0
NOISE_SIGMA = 0.00
DIST_MEAN = 0
DIST_SIGMA = 0

# Controller gains.
kp = 5
kd = 1
K = np.array([[0,  0],
              [kp, kd]])
C = np.eye(2)

# Operating point.
REF = np.array([np.pi, 0])


def disturbance(mean, sigma):
    ''' Normally distributed disturbance. '''
    return np.random.normal(mean, sigma)


def noise(mean, sigma):
    ''' Normally distributed sensor noise. '''
    return np.random.normal(mean, sigma)


def f(x, t, u):
    ''' State-update function: dx/dt = f(x). '''
    x1dot = x[1]
    x2dot = -GRAV / LENGTH * np.sin(x[0])
    return np.array([x1dot, x2dot]) + u


def step(x, dt, u):

    ''' Simulate the system for a single timestep. '''
    # Integrate the system over a single timestep.
    x = integrate.odeint(f, x, [0, dt], args=(u,))
    x = x[-1, :]

    # System output.
    y = np.dot(C, x)

    return x, y


def print_sim_time(t):
    if np.abs(t - int(t + 0.5)) < DT / 2.0:
        print('\rt = {}s'.format(t), end='')
        sys.stdout.flush()


def main():
    # Initialize the system.
    # Subtract operating point to put us into reference coordinates.
    x = X0 - REF
    y = np.dot(C, x)
    t = T0
    u = [0, 0]

    ts = np.array([t])
    ys = np.array([y])
    us = np.array([u])

    m1s = np.array([0])
    k1s = np.array([SIGNAL_SIGMA])

    m2s = np.array([0])
    k2s = np.array([SIGNAL_SIGMA])

    start_time = time.time()
    elapsed_time = np.array([0])

    gp1 = GaussianProcess(SEKernel, signal_sigma=SIGNAL_SIGMA)
    gp2 = GaussianProcess(SEKernel, signal_sigma=SIGNAL_SIGMA)

    # Simulate the system.
    while t < TF:
        # Predict the output.
        inp = [[x[0], x[1], u[1]]]
        m1, k1 = gp1.predict(inp)
        m2, k2 = gp2.predict(inp)

        # Simulate the system for one time step.
        x0 = x
        x, y = step(x0, DT, u)

        # Record what the output actually was.
        gp1.observe([[ x0[0], x0[1], u[1] ]], [[ y[0] ]])
        gp2.observe([[ x0[0], x0[1], u[1] ]], [[ y[1] ]])

        # Calculate control input for next step based on system output.
        u = np.dot(-K, y)

        t = t + DT

        # Record results.
        ts = np.append(ts, t)
        ys = np.append(ys, [y], axis=0)
        us = np.append(us, u)

        m1s = np.append(m1s, m1)
        k1s = np.append(k1s, np.sqrt(k1))

        m2s = np.append(m2s, m2)
        k2s = np.append(k2s, np.sqrt(k2))

        elapsed_time = np.append(elapsed_time, time.time() - start_time)

        print_sim_time(t)
    print()

    # Add operating point back to get back to normal coordinates.
    ys = ys + np.tile(REF, (ys.shape[0], 1))
    m1s = m1s + np.ones(m1s.shape[0]) * REF[0]

    # Plot the results.
    plt.figure(1)
    plt.subplot(211)
    plt.plot([T0, TF], [REF[0], REF[0]], label='Reference')
    plt.plot(ts, ys[:, 0], label='Actual')
    plt.plot(ts, m1s, label='Predicted')

    plot_sigma_bounds(ts, m1s, k1s, 3, (0.9,) * 3)
    plot_sigma_bounds(ts, m1s, k1s, 2, (0.8,) * 3)
    plot_sigma_bounds(ts, m1s, k1s, 1, (0.7,) * 3)

    plt.title('Inverted Pendulum')
    plt.ylabel('Angle (rad)')
    plt.legend()
    plt.grid()

    plt.subplot(212)
    plt.plot([T0, TF], [REF[1], REF[1]], label='Reference')
    plt.plot(ts, ys[:, 1], label='Actual')
    plt.plot(ts, m2s, label='Predicted')

    plot_sigma_bounds(ts, m2s, k2s, 3, (0.9,) * 3)
    plot_sigma_bounds(ts, m2s, k2s, 2, (0.8,) * 3)
    plot_sigma_bounds(ts, m2s, k2s, 1, (0.7,) * 3)

    plt.xlabel('Time (s)')
    plt.ylabel('Angular velocity (rad/s)')
    plt.legend()
    plt.grid()
    plt.show()

    # Plot real time vs. simulation time, so we can see how the required
    # computation increases as more data is observed over the course of the
    # simulation.
    plt.figure(2)
    plt.plot(ts, elapsed_time)
    plt.xlabel('Simulation time (s)')
    plt.ylabel('Real time (s)')
    plt.title('Real time vs. Simulation time')
    plt.grid()
    plt.show()

    # Plot angle uncertainty over a portion of the state-space, with input u
    # fixed at 0.
    print('Plotting sample of state-space...')
    fig = plt.figure(3)
    ax = fig.add_subplot(111, projection='3d')

    # Sample the GP across the state-space for u = 0.
    samples = np.mgrid[-2:2:0.1, -2:2:0.1, 0:1].reshape(3, -1).T
    m, cov = gp1.predict(samples)

    # Generate plotting values.
    X1, X2 = np.mgrid[-2:2:0.1, -2:2:0.1]
    Z = np.sqrt(np.diag(cov)).T.reshape(40, 40)

    ax.plot_surface(X1, X2, Z, cmap=cm.coolwarm, linewidth=0)
    plt.xlabel('Angle')
    plt.ylabel('Angular velocity')
    plt.show()


if __name__ == '__main__':
    main()
