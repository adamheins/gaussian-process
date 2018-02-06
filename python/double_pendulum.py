#!/usr/bin/env python3
# coding=utf-8
from __future__ import print_function

import sys

import numpy as np
import pygame

from control import System, SystemAnimation


# Physical pendulum parameters.
G = 9.81
L1 = L2 = 1
M1 = 10
M2 = 1
I1 = M1 * L1 ** 2
I2 = M2 * L2 ** 2

# Times are in seconds.
DT = 0.05
T0 = 0
TF = 20

# Initial condition.
X0 = np.array([np.pi * 0.25, 0, np.pi / 2, 0])

# Operating point.
REF = np.array([0, 0, 0, 0])


def tupadd(a, b):
    ''' Convenience functions for adding tuples element-wise. '''
    return tuple([x + y for x, y in zip(a, b)])


def f(x, t, r):
    ''' State-update function: dx/dt = f(x). '''
    # x = [th1, p1, th2, p2]
    c = np.cos(x[0] - x[2])
    s = np.sin(x[0] - x[2])

    th1dot = (L2*x[1] - L1*x[3]*c) / (L2*L1**2*(M1+M2-c**2))
    th2dot = (x[3] - M2*L1*L2*th1dot*c) / (M2*L2**2)
    p1dot  = -M2*L1*L2*th1dot*th2dot*s - (M1+M2)*G*L2*np.sin(x[0])
    p2dot  =  M2*L1*L2*th1dot*th2dot*s - M2*G*L2*np.sin(x[2])

    xdot = [th1dot, p1dot, th2dot, p2dot]
    return np.array(xdot)


def g(x, t, u):
    ''' System output. '''
    return x


def render_pendulum(screen, start_pos, length, angle, color):
    ''' Draw a single pendulum. '''
    x = int(length * np.sin(angle))
    y = int(length * np.cos(angle))
    end_pos = (start_pos[0] + x, start_pos[1] + y)

    pygame.draw.line(screen, color, start_pos, end_pos, 3)
    pygame.draw.circle(screen, color, end_pos, 10)

    return end_pos


def render(screen, sys):
    ''' Draw the whole system. '''
    length = 150
    start_pos = (250, 100)
    color = (0,) * 3

    # Draw past pendulum locations.
    pts1 = []
    pts2 = []
    for x in sys.xs:
        x1 = int(length * np.sin(x[0]))
        y1 = int(length * np.cos(x[0]))
        pt1 = tupadd(start_pos, (x1, y1))
        pts1.append(pt1)

        x2 = int(length * np.sin(x[2]))
        y2 = int(length * np.cos(x[2]))
        pt2 = tupadd(pt1, (x2, y2))
        pts2.append(pt2)

    pygame.draw.lines(screen, (0, 0, 255), False, pts1)
    pygame.draw.lines(screen, (255, 0, 0), False, pts2)

    # Draw the pendulums.
    pygame.draw.rect(screen, color, tupadd(start_pos, (-15, -8)) + (30, 16))
    end_pos = render_pendulum(screen, start_pos, length, sys.x[0], color)
    render_pendulum(screen, end_pos, length, sys.x[2], color)

    print_sim_time(sys.t)


def print_sim_time(t):
    if np.abs(t - int(t + 0.5)) < DT / 2.0:
        print('\rt = {:.1f}s'.format(round(t)), end='')
        sys.stdout.flush()


def main():
    sys = System(f, g, DT, X0, T0)
    anim = SystemAnimation(sys, (500, 500), (255,)*3)
    anim.animate(TF, lambda t: REF, render)
    print()


if __name__ == '__main__':
    main()
