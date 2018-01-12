import scipy.integrate as integrate
import numpy as np
import matplotlib.pyplot as plt


G = 9.81  # Gravity
L = 1     # Length
K = 1     # Damping


def u(t):
    ''' Control signal. '''
    return np.array([0, 0])


def f(t, x):
    ''' State-update function: dx/dt = f(x). '''
    x1dot = x[1]
    x2dot = - G / L * np.sin(x[0]) - K * x[1]
    return np.array([x1dot, x2dot])


def g(x):
    ''' Output function: y = g(x) '''
    return x


def main():
    x0 = np.array([np.pi/2, 0])

    # Times are in seconds.
    dt = 0.01
    t0 = 0
    t1 = 10

    # Setup numerical integrator with 4th order Runge-Kutta solver.
    solver = integrate.ode(f).set_integrator('dopri5')
    solver.set_initial_value(x0, t0)

    ts = np.array([t0])
    ys = np.array([g(x0)])

    # Simulate the system.
    while solver.successful() and solver.t + dt <= t1:
        t = solver.t + dt
        x = solver.integrate(t)
        y = g(x)

        # Record results.
        ts = np.append(ts, t)
        ys = np.append(ys, [y], axis=0)

    # Plot the results.
    plt.plot(ts, ys[:, 0], label='Angle')
    plt.plot(ts, ys[:, 1], label='Angular velocity')
    plt.xlabel('Time (s)')
    plt.ylabel('Outputs')
    plt.legend()
    plt.grid()
    plt.show()


if __name__ == '__main__':
    main()
