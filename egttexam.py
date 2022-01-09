#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
# Plotting libraries
import matplotlib.pylab as plt

import matplotlib.pyplot as plt
import egttools as egt

from egttools.analytical import replicator_equation
from egttools.utils import find_saddle_type_and_gradient_direction
from egttools.plotting import plot_gradient
from egttools.analytical import StochDynamics
from egttools.analytical import replicator_equation
from egttools.utils import find_saddle_type_and_gradient_direction
from egttools.plotting import plot_gradient


# In[27]:
from numpy import argmax

cost = 6.5


# In[28]:


import numpy as np

actions = np.asarray([cost, 0])

def is_the_same(type1,type2):
    return type1(0,0)==type2(0,0); #check if they are cooparating or not in round 0 so RM player or AD


# Let's estimate the payoffs through simulation

def always_defect(prev_contribution_group, t):
    return actions[1]


def r1(prev_contribution_group, t):
    """
    If prev_donations >= 1 -> coop
    else defect
    """
    if t == 0 or prev_contribution_group >= 1:
        return actions[0]  # coop
    else:
        return actions[1]  # defect
    
def r2(prev_contribution_group, t):
    """
    If prev_donations >= 2 -> coop
    else defect
    """
    if t == 0 or prev_contribution_group >= 2:
        return actions[0]  # coop
    else:
        return actions[1]  # defect

def r3(prev_contribution_group, t):
    """
    If prev_donations >= 3 -> coop
    else defect
    """
    if t == 0 or prev_contribution_group >= 3:
        return actions[0]  # coop
    else:
        return actions[1]  # defect

def r4(prev_contribution_group, t):
    """
    If prev_donations >= 4 -> coop
    else defect
    """
    if t == 0 or prev_contribution_group >= 4:
        return actions[0]  # coop
    else:
        return actions[1]  # defect
    
def r5(prev_contribution_group, t):
    """
    If prev_donations >= 5 -> coop
    else defect
    """
    if t == 0 or prev_contribution_group >= 5:
        return actions[0]  # coop
    else:
        return actions[1]  # defect

def npd(type1, type2, k, group_size, F, rounds):
    """
    Simulates a classical CRD (no timing uncertainty).
    :param F:
    :param cost:
    :param type1: [int] index of strategy 1
    :param type2: [int] index of strategy 2
    :param k: [int] number of group members adopting strategy 1
    :param group_size: [int] group size
    :param threshold: [int]
    :param r: [float] risk
    :param rounds: [int] total number of rounds
    :param endowment: [int] private endowment
    :return: [numpy.array] the payoffs of each strategy
    """
    payoffs = np.array([0, 0],dtype=float)
    if is_the_same(type1,type2):  # all members of the group adopt the same global strategy
        prev_contrib = group_size
        total=0
        for i in range(rounds):
            don = type1(prev_contrib, i)
            total+=(prev_contrib * don)
            #if payoffs[0] >= don1: #if endowment
            if don > 0:  # all coop
                payoffs[0] -= don
                payoffs[1] -= don
                               
        payoffs[0]+= (total*F)/group_size
        payoffs[1]+= (total*F)/group_size

    else:
        
        if(type1(k, 0)==0):
            prev_contrib= group_size-k
        else:
            prev_contrib = k
        prev_contrib-1
        total=0
        for i in range(rounds):  # the members of the group adopt different strategies
            don1 = type1(prev_contrib, i)
            don2 = type2(prev_contrib, i)
            don = max(don1,don2)
            total+=(prev_contrib * don)    
            # if payoffs[0] >= don1: if endowment
            payoffs[0] -= don1  
            # if payoffs[1] >= don2: if endowment
            payoffs[1] -= don2 

        payoffs[0]+= (total*F)/group_size
        payoffs[1]+= (total*F)/group_size

    return payoffs/rounds


class EstimatePayoffsNPD(object):
    strategies = ['R1','R2','R3','R4','R5''AL_DEFECT']
    strategies_caller = [r1,r2,r3,r4,r5,always_defect]
    ns = len(strategies)

    @staticmethod
    def estimate_payoff(invader, resident, group_size, F, rounds, iterations=100):
        """
        Estimates the payoff for invader and resident strategies,
        for the classical CRD.
        :param cost:
        :param F:
        :param invader: [int] index of the invading strategy
        :param resident: [int] index of the resident strategy
        :param group_size: [int] group size
        :param threshold: [int] contribution threshold for the conditional strategies
        :param r: [float] risk
        :param rounds: [int] number of rounds
        :param w: parameter not used
        :param iterations: [int] number of iterations used to average the game results
               (only relevant for stochastic results)
        :return: [lambda] function that returns the payoff of a crd group for each possible
                configuration of invader and resident strategies
        """
        payoffs = []
        for i in range(1, int(group_size) + 1):
            avg = 0.
            for _ in range(iterations):
                avg += npd(invader, resident, i, group_size, F, rounds)[0]
            payoffs.append(avg / float(iterations))

        # k is the number of invaders and z a dummy parameter
        return lambda k, z: payoffs[int(k) - 1]

    @staticmethod
    def estimate_payoffs(group_size, F, m0,
                         iterations=1000, save_name=None):
        """
        Estimates the payoffs of each strategy when playing against another.
        :param cost:
        :param F:
        :param group_size: [int] group size
        :param threshold: [int] contribution threshold for the conditional strategies
        :param r: [float] risk
        :param m0: [int] minimum number of rounds
        :param w: [float] probability that the game will and after the minimum number of rounds
        :param iterations: [int] number of iterations used to average the game results
        :param save_name: [string] name/path of file to save results, if None, the results are not saved
        :return: [numpy.array] the estimated payoffs
        """

        estimate = EstimatePayoffsNPD.estimate_payoff

        payoffs = np.asarray([[estimate(i, j, group_size, F, m0, iterations)
                                   for j in EstimatePayoffsNPD.strategies_caller] for i in
                                  EstimatePayoffsNPD.strategies_caller])

        return payoffs


# In[42]:


R=(1-0.9)**-1
F=4.25
N=5
estimateM1AD = EstimatePayoffsNPD()
payoffs=estimateM1AD.estimate_payoffs(N, F, round(R),10)


# In[43]:


# nb_strategies = 6; Z = 100;
# beta = 1
# test=np.array([
#     [payoffs[0][0](1,100),payoffs[0][1](1,100),payoffs[0][2](1,100),payoffs[0][3](1,100),payoffs[0][4](1,100),payoffs[0][5](1,100)],
#     [payoffs[1][0](1,100),payoffs[0][1](1,100),payoffs[1][2](1,100),payoffs[1][3](1,100),payoffs[1][4](1,100),payoffs[1][5](1,100)],
#     [payoffs[2][0](1,100),payoffs[2][1](1,100),payoffs[2][2](1,100),payoffs[2][3](1,100),payoffs[2][4](1,100),payoffs[2][5](1,100)],
#     [payoffs[3][0](1,100),payoffs[3][1](1,100),payoffs[3][2](1,100),payoffs[3][3](1,100),payoffs[3][4](1,100),payoffs[3][5](1,100)],
#     [payoffs[4][0](1,100),payoffs[4][1](1,100),payoffs[4][2](1,100),payoffs[4][3](1,100),payoffs[4][4](1,100),payoffs[4][5](1,100)],
#     [payoffs[5][0](1,100),payoffs[5][1](1,100),payoffs[5][2](1,100),payoffs[5][3](1,100),payoffs[5][4](1,100),payoffs[5][5](1,100)]
# ])
# print(test)
# evolver= StochDynamics(nb_strategies, test, Z)
# evolver.mu = 0
# stationary_SML = evolver.calculate_stationary_distribution(beta)
# transition_matrix,fixation_probabilities = evolver.transition_and_fixation_matrix(beta)
# stationary_distribution = egt.utils.calculate_stationary_distribution(transition_matrix)
# print(fixation_probabilities)
# print(stationary_distribution)


# In[44]:


# nb_strategies = 6; Z = 100; N = 5;
# beta = 1
# evolver= StochDynamics(nb_strategies, payoffs, Z,N)
# evolver.mu = 0
# stationary_SML = evolver.calculate_stationary_distribution(beta)
# transition_matrix,fixation_probabilities = evolver.transition_and_fixation_matrix(beta)
# stationary_distribution = egt.utils.calculate_stationary_distribution(transition_matrix)
# print(fixation_probabilities)
# print(stationary_distribution)


# In[45]:


# strategy_labels=["M1","M2","M3","M4","M5","AD"]
# ig, ax = plt.subplots(figsize=(5, 5), dpi=150)
# G = egt.plotting.draw_stationary_distribution(strategy_labels,
#                                               1/Z, fixation_probabilities, stationary_distribution,
#                                               node_size=600,
#                                               font_size_node_labels=8,
#                                               font_size_edge_labels=8,
#                                               font_size_sd_labels=8,
#                                               edge_width=1,
#                                               min_strategy_frequency=0.00001,
#                                               ax=ax)
# plt.axis('off')
# plt.show() # display


# In[46]:


# M2AD = np.array([
#     [payoffs[1][1](1,100), payoffs[1][5](1,100)],
#     [payoffs[5][1](1,100), payoffs[5][5](1,100)]
# ])
# nb_strategies = 2; Z = 100; N = 5;
# beta = 1
# evolverM2AD = StochDynamics(nb_strategies, M2AD, Z)
# evolverM2AD.mu = 0
# print("M2 invade AD with p =", evolverM2AD.fixation_probability(0,1,beta))
# print("AD invade M2 with p =",evolverM2AD.fixation_probability(1,0,beta))
# transition_matrix,fixation_probabilities = evolverM2AD.transition_and_fixation_matrix(beta)
# print()


# In[ ]:


# for F in [3.75,4.,4.25,4.5]:
#     payoffs=estimateM1AD.estimate_payoffs(N, F, round(R),10)
#     nb_strategies = 6; Z = 100; N = 5;
#     beta = 1
#     evolver= StochDynamics(nb_strategies, payoffs, Z,N)
#     evolver.mu = 0
#     transition_matrix,fixation_probabilities = evolver.transition_and_fixation_matrix(beta)
#     stationary_distribution = egt.utils.calculate_stationary_distribution(transition_matrix)
#     print(fixation_probabilities)
#     print(stationary_distribution)

Fs = np.arange(2.5,5.1,0.1)
rs = np.arange(2,42,2)

optimal = [[] for _ in range(len(rs))]

nb_strategies = 6;
Z = 100;
N = 5;
beta = 1

def fig3a(r):
    print(r)
    for f in Fs:
        payoffs = estimateM1AD.estimate_payoffs(N, f, round(rs[r]), 100)
        evolver = StochDynamics(nb_strategies, payoffs, Z, N)
        evolver.mu = 0
        transition_matrix, fixation_probabilities = evolver.transition_and_fixation_matrix(beta)
        stationary_distribution = egt.utils.calculate_stationary_distribution(transition_matrix)
        optimal[r].append(argmax(stationary_distribution))
    return (r,optimal[r])

labels = {0:"M=1",1:"M=2",2:"M=3",3:"M=4",4:"M=5",5:"AD"}
exist = {}
if __name__ == "__main__":
    from joblib import Parallel, delayed

    delayed_funcs = [delayed(fig3a)(i) for i in range(len(rs))]
    parallel_pool = Parallel(n_jobs=-1,backend="loky")
    result = parallel_pool(delayed_funcs)
    for r in result:
        optimal[r[0]] = r[1]


    # Plot stacked bar chart
    w = 2
    colors = ["lightgrey","grey","navy","blueviolet","darkviolet","slateblue"]
    for r in range(len(rs)):
        for o in range(len(optimal[r])):
            if not optimal[r][o] in exist:
                plt.bar(rs[r], [0.1], w,color=colors[optimal[r][o]],bottom=[Fs[o]],
                        label=labels[optimal[r][o]])
                exist[optimal[r][o]] = True
            else:
                plt.bar(rs[r], [0.1], w, color=colors[optimal[r][o]], bottom=[Fs[o]])
    plt.legend(loc=1)
    plt.xlabel('< r >')
    plt.ylabel('F')
    plt.show()


