# We consider different sampling policies

import numpy as np


def epsilonGreedy(state):

    Action_probabilities = np.ones(num_actions, dtype=float) * epsilon / num_actions

    best_action = np.argmax(Q[state])
    Action_probabilities[best_action] += 1.0 - epsilon

    return Action_probabilities

