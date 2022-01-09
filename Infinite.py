import math

import numpy as np
# Plotting libraries
import matplotlib.pylab as plt

from estimate_payoff import *

nb_points = 101
strategy_RM = np.linspace(0, 1, num=nb_points, dtype=np.float64)
strategy_AD = 1 - strategy_RM
states = np.array((strategy_RM, strategy_AD)).T



def RM_payoff(Nrm, M, c, R, F, N):
    if Nrm >= M: #player included
        return (((Nrm * c) * R * F) / N) - (c * R)  # R rounds
    else:
        return (((Nrm * c) * F) / N) - c  # 1 round



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

N = 5
c = 2
F = 3.5

sd = []
gx = []
ws = np.linspace(0.1,0.99,nb_points)
ex_ws = []
final = []
result = []
zero = np.zeros(shape=nb_points)
for m in range(2,N):
    final = []
    for w in ws:
        M = m
        R = (1 - w) ** -1

        # Calculate gradient
        G = np.array([rp(M, c, round(R), F, N, states[i]) for i in range(len(states))])

        #plt.plot(strategy_RM, G)

        idx = np.argwhere(np.diff(np.sign(G - zero))).flatten()
        #plt.plot(strategy_RM, G)
        #plt.plot(strategy_RM[idx], G[idx],'ro')
        #plt.show()
        idx = np.delete(idx,[0,len(idx)-1])
        for i in range(len(idx)):
            final.append((strategy_RM[idx[i]],w))

    final.sort(key=lambda y: y[0])
    sd = []
    ex_ws = []
    for i in final:
        sd.append(i[0])
        ex_ws.append(i[1])
    result.append((sd,ex_ws))

i=2
for s,e in result:
    plt.plot(s, e, label="M="+str(i))
    critical_m = min(e)
    plt.plot(strategy_RM,[critical_m for i in range(len(strategy_RM))],color="red",linestyle="dashed")
    i+=1

plt.legend(loc=3)
plt.xlabel("x")
plt.ylabel("w")
ax = plt.gca()
ax.set_ylim([0, 1])
plt.show()
