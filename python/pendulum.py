import scipy.integrate as integrate
import numpy as np
import matplotlib.pyplot as plt


grav = 9.81  # Gravity
length = 1     # Length
mass = 1     # Mass
inertia = mass * length ** 2

K = np.array([1, 1])

# Operating point.
REF = np.array([np.pi, 0])


def d(mean, sigma):
    ''' Normally distributed disturbance. '''
    return np.random.normal(mean, sigma)


def u(x):
    ''' Control signal. '''
    # Standard proportional controller.
    return -np.dot(K, g(x))


def f(t, x):
    ''' State-update function: dx/dt = f(x). '''
    x1dot = x[1]
    x2dot = -grav / length * np.sin(x[0])
    return np.array([x1dot, x2dot])


def g(x):
    ''' Output function: y = g(x) '''
    return x


def main():
    # Subtract operating point to put us into reference coordinates.
    x0 = np.array([np.pi / 2, 0]) - REF

    # Times are in seconds.
    dt = 0.01
    t0 = 0
    t1 = 10

    # Setup numerical integrator with 4th order Runge-Kutta solver.
    solver = integrate.ode(f).set_integrator('dopri5', nsteps=1000)
    solver.set_initial_value(x0, t0)

    ts = np.array([t0])
    ys = np.array([g(x0)])
    us = np.array([u(x0)])

    # Simulate the system.
    while solver.successful() and solver.t + dt <= t1:
        t = solver.t + dt
        x = solver.integrate(t)
        y = g(x)

        # u as function of y
        # add disturbance
        # set as new initial condition for the solver
        # would also be nice to blackbox x, only get y

        # Record results.
        ts = np.append(ts, t)
        ys = np.append(ys, [y], axis=0)
        us = np.append(us, u(x))

    # Add operating point back.
    ys = ys + np.tile(REF, (ys.shape[0], 1))

    # Plot the results.
    plt.plot(ts, ys[:, 0], label='Angle')
    plt.plot(ts, ys[:, 1], label='Angular velocity')
    plt.plot(ts, us, label='Torque')
    plt.xlabel('Time (s)')
    plt.ylabel('Outputs')
    plt.title('Inverted Pendulum')
    plt.legend()
    plt.grid()
    plt.show()


if __name__ == '__main__':
    main()
