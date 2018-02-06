import scipy.integrate as integrate
import numpy as np
import pygame


class System(object):
    def __init__(self, f, g, dt, x0, t0):
        self.f = f
        self.g = g
        self.dt = dt

        ref0 = np.zeros(x0.shape)
        y0 = g(x0, t0, ref0)

        self.x = x0
        self.y = y0
        self.t = t0
        self.ref = ref0

        self.ts = np.array([t0])
        self.refs = np.array([ref0])
        self.ys = np.array([y0])
        self.xs = np.array([x0])

    def step(self, ref, d=lambda x, u: np.zeros(x.shape)):
        # Integrate over a single timestep.
        x = integrate.odeint(self.f, self.x, [0, self.dt], args=(ref,))
        self.x = x[-1, :]

        # Output.
        self.y = self.g(self.x, self.t, ref)

        self.t += self.dt
        self.ref = ref

        # Record results.
        self.ts = np.append(self.ts, self.t)
        self.xs = np.append(self.xs, [self.x], axis=0)
        self.ys = np.append(self.ys, [self.y], axis=0)
        self.refs = np.append(self.refs, [ref], axis=0)

        return self.y


class SystemAnimation(object):
    def __init__(self, sys, size, bg_color):
        pygame.init()
        self.screen = pygame.display.set_mode(size)
        self.bg_color = bg_color
        self.sys = sys

        self.event_id = pygame.USEREVENT + 1
        pygame.time.set_timer(self.event_id, int(sys.dt * 1000))

    def animate(self, tf, ref_func, render_func):
        while self.sys.t < tf:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    break
                elif event.type == self.event_id:
                    self.sys.step(ref_func(self.sys.t))
                    self.screen.fill(self.bg_color)
                    render_func(self.screen, self.sys)
                    pygame.display.flip()
