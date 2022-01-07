#!/usr/bin/env python
# coding: utf-8

# In[270]:


import numpy as np
import random as random
from matplotlib import pyplot as plt


# In[311]:


beta = 1.
mu = 1e-3
Z=100
N=5
nb_generations=10**5
nb_runs=30

R=round((1-0.9)**-1)
c=6
F=4.25
#Cheater will be 0
#Mn will be n


# In[312]:


def select_random_without_replacement(population,number):
    return random.sample(list(enumerate(population)), number)

def estimate_fitness(selected, population, N, Z):
    
    PA=selected[0][1]
    PB=selected[1][1]
    
    numberRm=Z-population.count(0) 
    
    M5=population.count(5)
    M4=population.count(4)
    M3=population.count(3)
    M2=population.count(2)
    M1=population.count(1)
    
    if(numberRm>=5):
        investers=numberRm
        M=5
    elif(numberRm==4):
        investers=M4+M3+M2+M1
        M=4
    elif(numberRm==3):
        investers=M3+M2+M1
        M=3
    elif(numberRm==2):
        investers=M2+M1
        M=2
    elif(numberRm==1):
        M=1
        investers=M1
    else:
        M=0
        investers=0
        
    ResultA=payoffs[PA][PB](investers,Z)
    ResultB=payoffs[PB][PA](investers,Z)
    return ResultA,ResultB

def prob_imitation(beta,fitness):
    return 1./(1. + np.exp(beta*(fitness[0]-fitness[1])))


# In[313]:


def moran_step(current_state, beta, mu, N, Z):

    #This function implements a birth-death process over the population. 
    #At time t, two players are randomly selected from the population.
    strategies=[0,1,2,3,4,5]
    selected = select_random_without_replacement(current_state, 2)
    fitness = estimate_fitness(selected, current_state, N, Z)
    # Decide whether the player imitates
    if np.random.rand() < mu:
        current_state[selected[0][0]] = np.random.choice(strategies,size=1)[0]
    elif np.random.rand() < prob_imitation(beta,fitness):
        current_state[selected[0][0]] = current_state[selected[1][0]]

    return current_state


# In[314]:


def estimate_stationary_distribution(nb_runs, nb_generations, beta, mu,N, Z):
    counterAD=[]
    counterM1=[]
    counterM2=[]
    counterM3=[]
    counterM4=[]
    counterM5=[]
    for n in range(nb_runs):
        
        distribution = [random.random() for i in range(0,6)]
        total = sum(distribution)
        distribution = [ np.round((i/total)*100) for i in distribution]
        if(sum(distribution)==99):
            distribution[random.randint(0,len(distribution)-1)]+=1
        elif(sum(distribution)==101):
            distribution[random.randint(0,len(distribution)-1)]-=1
        distribution=[int(x) for x in distribution]
        base_population=[]
        i=0
        for entity in distribution:
            for num in range(entity):
                base_population.append(i)
            i+=1
        
        histo=[]
        histo.append(base_population)

        for j in range(nb_generations):
            next_step=moran_step(histo[j],beta,mu,N,Z)
            histo.append(next_step)
            nbAD=next_step.count(0)
            nbM1=next_step.count(1)
            nbM2=next_step.count(2)
            nbM3=next_step.count(3)
            nbM4=next_step.count(4)
            nbM5=next_step.count(5)
            #value=str(nbAD)+"|"+str(nbM1)+"|"+str(nbM2)+"|"+str(nbM3)+"|"+str(nbM4)+"|"+str(nbM5)
            #state[value]=state[value]=state.get(value,0)+1 
            counterAD.append(nbAD)
            counterM1.append(nbM1)
            counterM2.append(nbM2)
            counterM3.append(nbM3)
            counterM4.append(nbM4)
            counterM5.append(nbM5)
    
    
    return np.mean(counterAD),np.mean(counterM1),np.mean(counterM2),np.mean(counterM3),np.mean(counterM4),np.mean(counterM5)


# In[315]:


import numpy as np

actions = np.asarray([c, 0])

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


# In[306]:


R=(1-0.9)**-1
F=4.25
N=5
estimateM1AD = EstimatePayoffsNPD()
payoffs=estimateM1AD.estimate_payoffs(Z, F, round(R),100)


# In[316]:


counter=estimate_stationary_distribution(nb_runs,nb_generations,beta,mu,N,Z)


# In[317]:


print(counter)


# In[318]:


print(sum(counter))


# In[ ]:




