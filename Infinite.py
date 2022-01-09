import math

import numpy as np
# Plotting libraries
import matplotlib.pylab as plt

from egttools.analytical import replicator_equation
from egttools.utils import find_saddle_type_and_gradient_direction
from egttools.plotting import plot_gradient
from estimate_payoff import *

nb_points = 101
strategy_RM = np.linspace(0, 1, num=nb_points, dtype=np.float64)
W = np.linspace(0, 1, num=nb_points, dtype=np.float64)
strategy_AD = 1 - strategy_RM
states = np.array((strategy_RM, strategy_AD)).T

# Payoff matrix
F = 3.5
M = 4
N = 5
c = 2
w = 0.4
R = (1 - w) ** -1


def RM_payoff(Nrm, M, c, R, F, N):
    if Nrm+1 >= M:
        return (((Nrm+1 * c) * R * F) / N) - (c * R)  # R rounds
    else:
        return (((Nrm * c) * F) / N) - c  # 1 round


def AD_payoff(Nrm, M, c, R, F, N):
    if Nrm >= M:
        return ((((Nrm * c) * R) * F) / N) # R rounds
    else:
        return ((Nrm * c) * F) / N # 1 round


def fc(M, c, R, F, N, state):
    s = 0
    for j in range(N - 1):
        s += math.comb(N - 1, j) * (state[0] ** j) * ((state[1]) ** (N - 1 - j)) * RM_payoff(j + 1, M, c, R, F, N)
    return s


def fd(M, c, R, F, N, state):
    s = 0
    for j in range(N - 1):
        s += math.comb(N - 1, j) * (state[0] ** j) * ((state[1]) ** (N - 1 - j)) * AD_payoff(j, M, c, R, F, N)
    return s


def rp(M, c, R, F, N, state):
    x = state[0] * (state[1]) * (fc(M, c, R, F, N, state) - fd(M, c, R, F, N, state))
    return x

sd = []
ws = np.arange(0,1,0.1)
for w in ws:
    print(w)
    F = 3.5
    M = 2
    N = 5
    c = 2
    #w = 0.4
    R = (1 - w) ** -1

    # Calculate gradient
    G = np.array([rp(M, c, R, F, N, states[i]) for i in range(len(states))])

    # Find saddle points (where the gradient is 0)
    epsilon = 1e-7
    saddle_points_idx = np.where((G <= epsilon) & (G >= -epsilon))[0]
    saddle_points = saddle_points_idx / (nb_points - 1)
    sd.append(saddle_points)
    # Now let's find which saddle points are absorbing/stable and which aren't
    # we also annotate the gradient's direction among saddle poinst
    saddle_type, gradient_direction = find_saddle_type_and_gradient_direction(G, saddle_points_idx)

    ax = plot_gradient(strategy_RM,
                       G,
                       saddle_points,
                       saddle_type,
                       gradient_direction,
                       'Prisoner game replicator dynamics',
                       xlabel='$x$')
    plt.show()
print(sd)