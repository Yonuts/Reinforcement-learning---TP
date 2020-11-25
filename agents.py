import numpy as np
import random
random.seed(0)
np.random.seed(0)


"""
Contains the definition of the agent that will run in an
environment.
"""

class RandomAgent:
    def __init__(self):
        """Init a new agent.
        """

    def choose(self):
        """Acts in the environment.

        returns the chosen action.
        """
        return np.random.randint(0, 10)

    def update(self, action, reward):
        """Receive a reward for performing given action on
        given observation.

        This is where your agent can learn.
        """
        pass

# Easiest implementation
# But due to epsilon being fixed, after a lot of iterations of the algorithm, it doesn't stop exploring and it sometimes can give suboptimal arm choice.
class epsGreedyAgent:
    def __init__(self):
        self.A = [0,1,2,3,4,5,6,7,8,9]
        self.mu = {a:[] for a in self.A}
        self.epsilon = 0.1

    def choose(self):
        """Acts in the environment.

        returns the chosen action.
        """
        for a in self.A :
            if len(self.mu[a]) == 0 :
                return a

        if np.random.uniform(0,1) < self.epsilon :
            return np.random.randint(0,10)

        return np.argmax([np.mean(self.mu[a]) for a in self.A])

    def update(self, action, reward):
        """Receive a reward for performing given action on
        given observation.

        This is where your agent can learn.
        """
        self.mu[action].append(reward)

# Give a better evaluation of the average reward per arms for comparison between them.
class BesaAgent():
    # https://hal.archives-ouvertes.fr/hal-01025651v1/document
    def __init__(self):
        self.A = [0,1,2,3,4,5,6,7,8,9]
        self.mu = {a:[] for a in self.A}

    def choose(self):
        """Acts in the environment.

        returns the chosen action.
        """
        for a in self.A :
            if len(self.mu[a]) == 0 :
                return a
        
        best_a = 0
        for a in self.A :
          a_len = len(self.mu[a])
          best_len = len(self.mu[a_best])
          if a_len > best_len :
            if np.mean(np.random.choice(self.mu[a], best_len, replace=False)) > np.mean(self.mu[best_a]) :
              best_a = a
          else  :
            if np.mean(self.mu[a]) > np.mean(np.random.choise(self.mu[best_a], a_len, replace=False)) :
              best_a = a
        return best_a

    def update(self, action, reward):
        """Receive a reward for performing given action on
        given observation.

        This is where your agent can learn.
        """
        self.mu[a].append(reward)

# Transform the average reward of each arm into a probability distribution which will be used to draw an arm choice.
# The more we converge to a solution, the less we do exploration.
class SoftmaxAgent:
    # https://www.cs.mcgill.ca/~vkules/bandits.pdf
    def __init__(self):
        self.A = [0,1,2,3,4,5,6,7,8,9]
        self.t = 1

    def choose(self):
        """Acts in the environment.

        returns the chosen action.
        """
        for a in self.A :
            if len(self.mu[a]) == 0 :
                return a

        mean_vector = [np.mean(self.mu[a]) for a in self.A]/self.t
        # compute the softmax
        softmax_prob = np.exp(mean_vector)/np.sum(mean_vector, axis=0)
        # draw an action from the softmax probability distribution
        return np.random.choice(self.A,1,p=softmax_prob)

    def update(self, action, reward):
        """Receive a reward for performing given action on
        given observation.

        This is where your agent can learn.
        """
        self.mu[a].append(reward)

# Give priority to explore arms that were not explored enough (as the less an arm is explored, the more chance it has to be explored)
# but it takes longer for the algorithm to converge as we prefer to have a good amount of explorations.
class UCBAgent:
    # https://homes.di.unimi.it/~cesabian/Pubblicazioni/ml-02.pdf
    def __init__(self):
        self.A = [0,1,2,3,4,5,6,7,8,9]
        self.mu = {a:[] for a in self.A}
        self.n = [0 for a in self.A]
        self.t = 0

    def choose(self):
        """Acts in the environment.

        returns the chosen action.
        """
        for a in self.A :
            if len(self.mu[a]) == 0 :
                return a

        return np.argmax([np.mean(self.mu[a])+np.sqrt(2*np.log(self.t)/self.n[a]) for a in self.A])

    def update(self, a, r):
        """Receive a reward for performing given action on
        given observation.

        This is where your agent can learn.
        """
        self.mu[a].append(r)
        self.n[a] += 1
        self.t += 1

# Determine for each arm a probability distribution of the parameter theta by using a Bayesian approach, we can then sample theta and choose the best one.
class ThompsonAgent:
    # https://en.wikipedia.org/wiki/Thompson_sampling
    def __init__(self):
        self.A = [0,1,2,3,4,5,6,7,8,9]
        self.trials = [0 for a in self.A]
        self.success = [0 for a in self.A]

    def choose(self):
        """Acts in the environment.

        returns the chosen action.
        """
        for a in self.A :
            if len(self.mu[a]) == 0 :
                return a
        
        thetas = [0 for a in self.A] 
        # we choose a uniform distribution for the prior distribution
        for a in self.A :
            thetas[a] = np.random.beta(1+self.success[a],1+self.trials[a]-self.success[a])
        return np.argmax(thetas)

    def update(self, a, r):
        """Receive a reward for performing given action on
        given observation.

        This is where your agent can learn.
        """
        self.mu[a].append(r)
        self.trials[a] += 1
        if r > 0 :
          self.succces[a] += 1


class KLUCBAgent:
    # See: https://hal.archives-ouvertes.fr/hal-00738209v2
    def __init__(self):
        self.A = [0,1,2,3,4,5,6,7,8,9]

    def choose(self):
        """Acts in the environment.

        returns the chosen action.
        """
        raise NotImplemented

    def update(self, a, r):
        """Receive a reward for performing given action on
        given observation.

        This is where your agent can learn.
        """
        raise NotImplemented
