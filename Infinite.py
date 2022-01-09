import math

import numpy as np
# Plotting libraries
import matplotlib.pylab as plt

from estimate_payoff import *

nb_points = 101
strategy_RM = np.linspace(0, 1, num=nb_points, dtype=np.float64)
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
    if Nrm+1 >= M: #player included
        return ((((Nrm+1 * c) * R * F) / N) - (c * R))  # R rounds
    else:
        return ((((Nrm * c) * F) / N) - c)  # 1 round


def AD_payoff(Nrm, M, c, R, F, N):
    if Nrm >= M:
        return ((((Nrm * c) * R) * F) / N) # R rounds
    else:
        return (((Nrm * c) * F) / N) # 1 round


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
gx = []
ws = np.linspace(0.1,0.99,5)
ex_ws = []
final = []
zero = np.zeros(shape=nb_points)

for w in ws:
    F = 3.5
    M = 5
    N = 5
    c = 2
    #w = 0.4
    R = (1 - w) ** -1

    # Calculate gradient
    G = np.array([rp(M, c, round(R), F, N, states[i]) for i in range(len(states))])


    idx = np.argwhere(np.diff(np.sign(G - zero))).flatten()
    plt.plot(strategy_RM, G,label="w="+str(w))

plt.plot(strategy_RM, zero, color="black")
plt.legend(loc=3)
plt.xlabel('x')
plt.ylabel('G(x)')
plt.show()