import numpy as np

actions = np.asarray([0, 1])


# Let's estimate the payoffs through simulation

def always_defect(prev_contribution_group, threshold, t):
    return actions[1]


def rm(prev_contribution_group, threshold, t):
    """
    If prev_donations >= threshold -> contribute 4
    else contribute 0
    """
    if t == 0:
        return actions[0]
    elif prev_contribution_group < threshold:
        return actions[0]
    else:
        return actions[2]


def crd(type1, type2, k, group_size, target, threshold, r, rounds, endowment):
    """
    Simulates a classical CRD (no timing uncertainty).
    :param type1: [int] index of strategy 1
    :param type2: [int] index of strategy 2
    :param k: [int] number of group members adopting strategy 1
    :param group_size: [int] group size
    :param target: [int]
    :param threshold: [int]
    :param r: [float] risk
    :param rounds: [int] total number of rounds
    :param endowment: [int] private endowment
    :return: [numpy.array] the payoffs of each strategy
    """
    donations = 0.
    contributions1, contributions2 = 0., 0.
    public_account = 0.
    payoffs = np.array([endowment, endowment])

    if type1 == type2:  # all members of the group adopt the same strategy
        for i in range(rounds):
            don = type1(donations - contributions1, threshold, i)
            donations = 0

            if payoffs[0] >= don:
                payoffs[0] -= don
                payoffs[1] -= don
                donations += group_size * don

            public_account += donations
            if public_account >= target:
                return payoffs
    else:
        for i in range(rounds):  # the members of the group adopt different strategies
            contributions1 = type1(donations - contributions1, threshold, i)
            contributions2 = type2(donations - contributions2, threshold, i)
            donations = 0

            if payoffs[0] >= contributions1:
                payoffs[0] -= contributions1
                donations += k * contributions1

            if payoffs[1] >= contributions2:
                payoffs[1] -= contributions2
                donations += (group_size - k) * contributions2

            public_account += donations
            if public_account >= target:
                return payoffs

    # if the target isn't met, the expected payoff is (1-risk) * payoff
    return (1.0 - r) * payoffs


def crd_uncertainty(type1, type2, frac, group_size, target, threshold, r, m0, w, endowment):
    """
    CRD with timing uncertainty.
    The final round is sampled from a geometric distribution with probability :param w
    summing a minimum number of rounds :param m0.
    """
    # we have to subtract 1 so that 1 trial means the game ends in m0 round
    total_rounds = np.random.geometric(p=w, size=1) + m0 - 1
    return crd(type1, type2, frac, group_size, target, threshold, r, int(total_rounds), endowment)


class EstimatePayoffsCRD(object):
    strategies = ['AL0', 'AL2', 'AL4', 'COMPENSATOR', 'RECIPROCAL']
    strategies_caller = [always_zero, always_two, always_four, compensator, reciprocal]
    ns = len(strategies)

    @staticmethod
    def estimate_payoff(invader, resident, group_size, target, threshold, r, rounds, w, endowment, iterations=100):
        """
        Estimates the payoff for invader and resident strategies,
        for the classical CRD.
        :param invader: [int] index of the invading strategy
        :param resident: [int] index of the resident strategy
        :param group_size: [int] group size
        :param target: [int] collective target
        :param threshold: [int] contribution threshold for the conditional strategies
        :param r: [float] risk
        :param rounds: [int] number of rounds
        :param w: parameter not used
        :param endowment: [int] private endowment
        :param iterations: [int] number of iterations used to average the game results
               (only relevant for stochastic results)
        :return: [lambda] function that returns the payoff of a crd group for each possible
                configuration of invader and resident strategies
        """
        payoffs = []
        for i in range(1, int(group_size) + 1):
            avg = 0.
            for _ in range(iterations):
                avg += crd(invader, resident, i, group_size, target, threshold, r, rounds, endowment)[0]
            payoffs.append(avg / float(iterations))

        # k is the number of invaders and z a dummy parameter
        return lambda k, z: payoffs[int(k) - 1]

    @staticmethod
    def estimate_payoff_uncertainty(invader, resident, group_size, target, threshold, r, m0, w, endowment,
                                    iterations=100):
        """
        Estimates the payoff for invader and resident strategies for the CRD with timing uncertainty
        :param invader: [int] index of the invading strategy
        :param resident: [int] index of the resident strategy
        :param group_size: [int] group size
        :param target: [int] collective target
        :param threshold: [int] contribution threshold for the conditional strategies
        :param r: [float] risk
        :param m0: [int] minimum number of rounds
        :param w: [float] probability that the game will and after the minimum number of rounds
        :param endowment: [int] private endowment
        :param iterations: [int] number of iterations used to average the game results
        :return: [lambda] function that returns the payoff of a crd group for each possible
                configuration of invader and resident strategies
        """
        payoffs = []
        for i in range(1, int(group_size) + 1):
            avg = 0.
            for _ in range(iterations):
                avg += crd_uncertainty(invader, resident, i, group_size, target, threshold, r, m0, w, endowment)[0]
            payoffs.append(avg / float(iterations))

        # k is the number of invaders and z a dummy parameter
        return lambda k, z: payoffs[int(k) - 1]

    @staticmethod
    def estimate_payoffs(group_size, target, threshold, r, m0, w, endowment,
                         iterations=1000, uncertainty=False, save_name=None):
        """
        Estimates the payoffs of each strategy when playing against another.
        :param group_size: [int] group size
        :param target: [int] collective target
        :param threshold: [int] contribution threshold for the conditional strategies
        :param r: [float] risk
        :param m0: [int] minimum number of rounds
        :param w: [float] probability that the game will and after the minimum number of rounds
        :param endowment: [int] private endowment
        :param iterations: [int] number of iterations used to average the game results
        :param uncertainty: [boolean] indicates whether the simulation is with or without timing uncertainty
        :param save_name: [string] name/path of file to save results, if None, the results are not saved
        :return: [numpy.array] the estimated payoffs
        """
        import dill as pickle

        try:
            payoffs = pickle.load(open("{}.pkl".format(save_name), "rb"))
        except IOError:
            if uncertainty:
                estimate = EstimatePayoffsCRD.estimate_payoff_uncertainty
            else:
                estimate = EstimatePayoffsCRD.estimate_payoff

            payoffs = np.asarray([[estimate(i, j, group_size, target, threshold, r, m0, w, endowment, iterations)
                                   for j in EstimatePayoffsCRD.strategies_caller] for i in
                                  EstimatePayoffsCRD.strategies_caller])

            if save_name is not None:
                pickle.dump(payoffs, open("{}.pkl".format(save_name), "wb"))

        return payoffs