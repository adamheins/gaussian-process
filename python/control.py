import scipy.integrate as integrate
import numpy as np
import matplotlib.pyplot as plt


A = np.eye(2)
B = np.zeros((2, 2))
C = np.eye(2)


def u(t):
    ''' Control signal. '''
    return np.array([0, 0])


def f(t, x):
    ''' State-update function: dx/dt = f(x). '''
    return np.dot(A, x) + np.dot(B, u(t))


def g(x):
    ''' Output function: y = g(x) '''
    return np.dot(C, x)


def main():
    x0 = np.array([0, 0])

    dt = 0.1
    t0 = 0
    t1 = 1

    # Setup numerical integrator with 4th order Runge-Kutta solver.
    solver = integrate.ode(f).set_integrator('dopri5')
    solver.set_initial_value(x0, t0)

    ts = [t0]
    ys = [g(x0)]

    # Simulate the system.
    while solver.successful() and solver.t + dt <= t1:
        t = solver.t + dt
        x = solver.integrate(t)
        y = g(x)

        # Record results.
        ts.append(t)
        ys.append(y)

    # Plot the results.
    plt.plot(ts, ys)
    plt.xlabel('Time (s)')
    plt.ylabel('Outputs')
    plt.show()


if __name__ == '__main__':
    main()


