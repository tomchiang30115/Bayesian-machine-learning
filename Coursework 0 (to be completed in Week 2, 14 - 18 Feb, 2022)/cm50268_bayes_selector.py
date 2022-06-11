#
# BayesXXXXSelector
#
# K-Arm bandit selector classes, using Thompson sampling
#
# Currently implemented for:
#
# - Bernoulli
# - Gamma
# - Gaussian
#
import numpy as np
from scipy import stats
import cm50268_bandit as lab0


class BayesBernoulliSelector(lab0.Selector):
    def __init__(self, K, alpha, beta):
        super().__init__(K)
        # Store fixed values
        self._alpha = alpha
        self._beta = beta
        # Declare state
        self._n1 = None
        self._n0 = None
        self._Alpha_N = None
        self._Beta_N = None
        #
        self.reset_state()

    def reset_state(self):
        self._n1 = np.zeros(self.K)
        self._n0 = np.zeros(self.K)
        self._Alpha_N = np.full(self.K, self._alpha)
        self._Beta_N = np.full(self.K, self._beta)

    def lever_select(self):
        theta = np.empty(self.K)
        for idx in range(self.K):
            # Sample from the Beta posterior
            # for every arm
            theta[idx] = stats.beta.rvs(a=self._Alpha_N[idx], b=self._Beta_N[idx])
        # Choose highest sample
        lever = np.argmax(theta)
        return lever

    def update_state(self, lever, payout):
        # This could be done more efficiently, but we leave the posterior update in
        # its explicit form for now
        if payout > 0:
            # Success!
            self._n1[lever] += 1
        else:
            # Failure :-(
            self._n0[lever] += 1
        #
        self._Alpha_N[lever] = self._alpha + self._n1[lever]
        self._Beta_N[lever] = self._beta + self._n0[lever]


class BayesGammaSelector(lab0.Selector):
    def __init__(self, K, alpha_0, beta_0, alpha_true):
        super().__init__(K)
        self._alpha_true = alpha_true
        self._alpha_0 = alpha_0
        self._beta_0 = beta_0
        #
        self._Alpha_N = None
        self._Beta_N = None
        self.reset_state()

    def reset_state(self):
        self._Alpha_N = np.full(self.K, self._alpha_0)
        self._Beta_N = np.full(self.K, self._beta_0)

    def lever_select(self):
        #
        beta = np.empty(self.K)
        for idx in range(self.K):
            # Sample from the Gamma posterior Gamma(a_n, b_n) for every arm
            # Note that stats.gamma only explicitly takes alpha as a parameter,
            # beta must be passed as "scale" reciprocated
            beta[idx] = stats.gamma.rvs(a=self._Alpha_N[idx], scale=1/self._Beta_N[idx])
        # Choose SMALLEST sample (since mean return is alpha / beta)
        lever = np.argmin(beta)
        return lever

    def update_state(self, lever, payout):
        self._Alpha_N[lever] += self._alpha_true
        self._Beta_N[lever] += payout


class BayesGaussSelector(lab0.Selector):
    def __init__(self, K, mu_0, s2_0, s2_true):
        #
        super().__init__(K)
        self._s2_true = s2_true
        self._mu_0 = mu_0
        self._s2_0 = s2_0
        # State
        self._Totals = None
        self._Counts = None
        self._Mu_N = None
        self._S2_N = None
        #
        self.reset_state()

    def lever_select(self):
        #
        Y = np.zeros(self.K)
        for idx in range(self.K):
            # Sample from the Gaussian posterior N(mu_n, s2_n)
            # for every arm
            Y[idx] = stats.norm.rvs(loc=self._Mu_N[idx],
                                    scale=np.sqrt(self._S2_N[idx]))
        # Choose highest sample
        lever = np.argmax(Y)
        return lever

    def reset_state(self):
        self._Totals = np.zeros(self.K)
        self._Counts = np.zeros(self.K)
        self._Mu_N = np.full(self.K, self._mu_0)
        self._S2_N = np.full(self.K, self._s2_0)

    def update_state(self, lever, payout):
        # Maintain running averages
        self._Counts[lever] += 1
        self._Totals[lever] += payout
        N = self._Counts[lever]  # for clarity
        xbar = self._Totals[lever] / N
        #
        fac = N * self._s2_0 / (N * self._s2_0 + self._s2_true)
        self._Mu_N[lever] = fac * xbar + (1.0 - fac) * self._mu_0
        self._S2_N[lever] = 1.0 / (N / self._s2_true + 1.0 / self._s2_0)
