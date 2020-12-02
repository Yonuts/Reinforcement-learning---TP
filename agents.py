import numpy as np
import environment
from sklearn.neural_network import MLPRegressor

"""
Contains the definition of the agent that will run in an
environment.
"""


class Q_Learning_Function_Approximation:
    """ Q-Learning with Function Approximation
    """

    def __init__(self):
        """Init a new agent.
        """
        # initial fit with terminal state, for any value of velocity and action, Q(s,a) = 0
        self.model = MLPRegressor(random_state=1).fit([[0.5, e, 0] for e in np.random.uniform(-0.07, 0.07, 30)], [0 for i in range(30)])
        self.model.partial_fit([[0.5, e, 1] for e in np.random.uniform(-0.07, 0.07, 30)], [0 for i in range(30)])
        self.model.partial_fit([[0.5, e, 2] for e in np.random.uniform(-0.07, 0.07, 30)], [0 for i in range(30)])
        self.gamma = 0.95

    def act(self, state):
        """Acts given an observation of the environment.

        Takes as argument an observation of the current state, and
        returns the chosen action.
        See environment documentation: https://github.com/openai/gym/wiki/MountainCar-v0
        Possible action: [0, 1, 2]
        Range observation (tuple):
            - position: [-1.2, 0.6]
            - velocity: [-0.07, 0.07]
        """
        return np.random.choice([0, 1, 2])

    def update(self, state, action, reward, new_state, terminal):
        """Receive a reward for performing given action.

        This is where your agent can learn. (Build model to approximate Q(s, a))
        Parameters:
            state: current state
            action: action done in state
            reward: reward received after doing action in state
            new_state: next state
            terminal: boolean if new_state is a terminal state or not
        """
        if not terminal:
            q_value = reward
            q_table = [self.model.predict(np.array([np.concatenate((new_state, [i]))]))[0] for i in range(3)]
            q_value += self.gamma * max(q_table)
            self.model.partial_fit([np.concatenate((state, [action]))], [q_value])

    def q(self, state, action):
        """Final Q function. It will be used for visualization purposes.
        Parameters:
            state: vector in R^2
            action: scalar (0, 1 or 2)
        Return:
            Value (scalar) of Q(state, action)
        """
        return self.model.predict([np.concatenate((state, [action]))])[0]


class Double_Q_Learning:
    """ Q-Learning with Function Approximation
    """

    def __init__(self):
        """Init a new agent.
        """
        pass

    def act(self, state):
        """Acts given an observation of the environment.

        Takes as argument an observation of the current state, and
        returns the chosen action.
        See environment documentation: https://github.com/openai/gym/wiki/MountainCar-v0
        Possible action: [0, 1, 2]
        Range observation (tuple):
            - position: [-1.2, 0.6]
            - velocity: [-0.07, 0.07]
        """
        return [np.random.choice([0, 1, 2])]

    def update(self, state, action, reward, new_state, terminal):
        """Receive a reward for performing given action.

        This is where your agent can learn. (Build model to approximate Q(s, a))
        Parameters:
            state: current state
            action: action done in state
            reward: reward received after doing action in state
            new_state: next state
            terminal: boolean if new_state is a terminal state or not
        """
        pass

    def q(self, state, action):
        """Final Q function. It will be used for visualization purposes.
        Parameters:
            state: vector in R^2
            action: scalar (0, 1 or 2)
        Return:
            Value (scalar) of Q(state, action)
        """
        return np.random.uniform(0, 1)
