import sys


def print_sim_time(t, dt):
    if abs(t - int(t + 0.5)) < dt / 2.0:
        print('\rt = {:.1f}s'.format(round(t)), end='')
        sys.stdout.flush()


def tupadd(a, b):
    ''' Convenience functions for adding tuples element-wise. '''
    return tuple([x + y for x, y in zip(a, b)])
