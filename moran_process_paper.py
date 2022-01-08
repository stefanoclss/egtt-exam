#!/usr/bin/env python
# coding: utf-8

# In[569]:


import numpy as np
import random as random
from matplotlib import pyplot as plt


# In[637]:


beta = 0.05
mu = 1e-3
Z=100
N=5
nb_generations=10**4
nb_runs=30

R=round((1-0.9)**-1)
c=6
F=4.25
#Cheater will be 0
#Mn will be n


# In[638]:


def select_random_without_replacement(population,number):
    return random.sample(list(enumerate(population)), number)

def estimate_fitness(selected, population, N, Z):
    
    PA=selected[0][1]
    PB=selected[1][1]
    
    groups=select_random_without_replacement(population,N)
    group=[]
    for player in groups:
        group.append(player[1])
    numberRm=N-group.count(0) 
    
    M5=group.count(5)
    M4=group.count(4)
    M3=group.count(3)
    M2=group.count(2)
    M1=group.count(1)
    
    investers = numberRm
    
    ResultA=payoffs[PA][PB](investers,N)
    ResultB=payoffs[PB][PA](investers,N)
    return ResultA,ResultB

def prob_imitation(beta,fitness):
    return 1./(1. + np.exp(beta*(fitness[0]-fitness[1])))


# In[639]:


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


# In[640]:


def estimate_stationary_distribution(nb_runs, nb_generations, beta, mu,N, Z):
    counterAD=[]
    counterM1=[]
    counterM2=[]
    counterM3=[]
    counterM4=[]
    counterM5=[]
    for n in range(nb_runs):
        print("run",n)
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
    
    
    return np.mean(counterAD)/100,np.mean(counterM1)/100,np.mean(counterM2)/100,np.mean(counterM3)/100,np.mean(counterM4)/100,np.mean(counterM5)/100


# In[641]:


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
    strategies = ['R5''AL_DEFECT','R1','R2','R3','R4']
    strategies_caller = [always_defect,r1,r2,r3,r4,r5]
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


# In[642]:


R=(1-0.9)**-1
F=4.25
N=5
estimateM1AD = EstimatePayoffsNPD()
payoffs=estimateM1AD.estimate_payoffs(N, F, round(R),100)


# In[643]:


counter=estimate_stationary_distribution(nb_runs,nb_generations,beta,mu,N,Z)


# In[644]:


print(counter)


# In[645]:


print(sum(counter))


# In[631]:


values=[10**-4,2*10**-4,4*10**-4,6*10**-4,8*10**-4,10**-3,2*10**-3,4*10**-3,6*10**-3,8*10**-3,10**-2,2*10**-2,4*10**-2,6*10**-2,8*10**-2,10**-1,2*10**-1,4*10**-1,6*10**-1,8*10**-1,1]
values = np.logspace(10**-4,1, num=40, endpoint=True, base=10.0)

# In[648]:


values = np.exp(np.linspace(np.log(10**-4), np.log(1), 20))


# In[649]:


print(values)


# In[650]:


StationaryAD=[]
StationaryM1=[]
StationaryM2=[]
StationaryM3=[]
StationaryM4=[]
StationaryM5=[]
StationaryM6=[]

for mus in values:
    counter=estimate_stationary_distribution(nb_runs,nb_generations,beta,mus,N,Z)
    StationaryAD.append(counter[0])
    StationaryM1.append(counter[1])
    StationaryM2.append(counter[2])
    StationaryM3.append(counter[3])
    StationaryM4.append(counter[4])
    StationaryM5.append(counter[5])


# In[651]:


plt.scatter(values,StationaryM1,color="black")
plt.scatter(values,StationaryM2,marker="s")
plt.scatter(values,StationaryM3,marker="D")
plt.scatter(values,StationaryM4,marker="^")
plt.scatter(values,StationaryM5,marker="*")
plt.scatter(values,StationaryAD,marker="v")
plt.xscale("log")
plt.legend(["M1","M2","M3","M4","M5","AD"])
plt.show()


# In[ ]:




