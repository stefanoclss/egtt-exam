import numpy as np

cost = 2

actions = np.asarray([cost, 0])


# Let's estimate the payoffs through simulation

def always_defect(prev_contribution_group, threshold, t):
    return actions[1]


def rm(prev_contribution_group, threshold, t):
    """
    If prev_donations >= threshold -> coop
    else defect
    """
    if t == 0 or prev_contribution_group >= threshold:
        return actions[0]  # coop
    else:
        return actions[1]  # defect


def npd(type1, type2, k, group_size, F, threshold, rounds):
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
    payoffs = np.array([0, 0], dtype=float)
    if type1 == type2:  # all members of the group adopt the same global strategy
        prev_contrib = group_size
        total = 0
        for i in range(rounds):
            don = type1(prev_contrib,threshold, i)
            total += (prev_contrib * don)
            # if payoffs[0] >= don1: #if endowment
            if don > 0:  # all coop
                payoffs[0] -= don
                payoffs[1] -= don

        payoffs[0] += (total * F) / group_size
        payoffs[1] += (total * F) / group_size

    else:

        if (type1(k, threshold, 0) == 0):
            prev_contrib = group_size - k
        else:
            prev_contrib = k
        prev_contrib - 1
        total = 0
        for i in range(rounds):  # the members of the group adopt different strategies
            don1 = type1(prev_contrib,threshold, i)
            don2 = type2(prev_contrib,threshold, i)
            don = max(don1, don2)
            total += (prev_contrib * don)
            # if payoffs[0] >= don1: if endowment
            payoffs[0] -= don1
            # if payoffs[1] >= don2: if endowment
            payoffs[1] -= don2

        payoffs[0] += (total * F) / group_size
        payoffs[1] += (total * F) / group_size

    return payoffs / rounds


class EstimatePayoffsNPD(object):
    strategies = ['RM','AL_DEFECT',]
    strategies_caller = [rm , always_defect]
    ns = len(strategies)

    @staticmethod
    def estimate_payoff(invader, resident, group_size, F, threshold, rounds, iterations=100):
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
                avg += npd(invader, resident, i, group_size, F, threshold, rounds)[0]
            payoffs.append(avg / float(iterations))

        # k is the number of invaders and z a dummy parameter
        return lambda k, z: payoffs[int(k) - 1]

    @staticmethod
    def estimate_payoffs(group_size, F, threshold, m0,
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
        import dill as pickle

        try:
            payoffs = pickle.load(open("{}.pkl".format(save_name), "rb"))
        except IOError:
            estimate = EstimatePayoffsNPD.estimate_payoff

            payoffs = np.asarray([[estimate(i, j, group_size, F, threshold, m0, iterations)
                                   for j in EstimatePayoffsNPD.strategies_caller] for i in
                                  EstimatePayoffsNPD.strategies_caller])

            if save_name is not None:
                pickle.dump(payoffs, open("{}.pkl".format(save_name), "wb"))

        return payoffs
