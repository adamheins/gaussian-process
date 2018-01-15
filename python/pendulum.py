#!/usr/bin/env python3
# coding=utf-8

import scipy.integrate as integrate
import numpy as np
import matplotlib.pyplot as plt


grav = 9.81
length = 1
mass = 1
inertia = mass * length ** 2

K = np.array([1, 1])

# Operating point.
REF = np.array([np.pi, 0])


def disturbance(mean, sigma):
    ''' Normally distributed disturbance. '''
    return np.random.normal(mean, sigma)


def control(y):
    ''' Control signal. '''
    # Standard proportional controller.
    return -np.dot(K, y)


def f(x, t, u):
    ''' State-update function: dx/dt = f(x). '''
    x1dot = x[1]
    x2dot = -grav / length * np.sin(x[0]) + u
    return np.array([x1dot, x2dot])


def out(x):
    ''' Output function: y = g(x) '''
    return x


def step(x, u, d, dt):
    x = integrate.odeint(f, x, [0, dt], args=(u + d,))
    x = x[1, :]
    y = out(x)
    return x, y


def main():
    # Subtract operating point to put us into reference coordinates.
    x0 = np.array([np.pi / 2, 0]) - REF

    # Times are in seconds.
    dt = 0.01
    t0 = 0
    t1 = 10

    # Initialize the system.
    x = x0
    y = out(x0)
    t = t0
    u = 0

    ts = np.array(t)
    ys = np.array([y])
    us = np.array(u)

    # Simulate the system.
    while t <= t1:
        # Generate disturbance.
        d = disturbance(0, 0.1)

        # Simulate the system for one time step.
        x, y = step(x, u, d, dt)

        # Calculate control input for next step based on system output.
        u = control(y)

        t = t + dt

        # Record results.
        ts = np.append(ts, t)
        ys = np.append(ys, [y], axis=0)
        us = np.append(us, u)

    # Add operating point back to get back to normal coordinates.
    ys = ys + np.tile(REF, (ys.shape[0], 1))

    # Plot the results.
    plt.plot([0, t1], [REF[0], REF[0]], label='Reference Angle (rad)')
    plt.plot(ts, ys[:, 0], label='Angle (rad)')
    plt.plot(ts, ys[:, 1], label='Angular Velocity (rad/s)')
    plt.plot(ts, us, label=u'Applied Torque (N·m)')

    plt.xlabel('Time (s)')
    plt.ylabel('Signals')
    plt.title('Inverted Pendulum')
    plt.legend()
    plt.grid()
    plt.show()


if __name__ == '__main__':
    main()
