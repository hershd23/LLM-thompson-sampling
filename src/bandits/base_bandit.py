import numpy as np

class BaseBandit(object):
    """
    Base class for all the bandit algorithms
    """

    def __init__(self):
        """Initializes the bandit algorithm"""
        raise NotImplementedError

    def update_observations(self, arm, reward):
        """Updates the observation for the bandit algorithm"""
        raise NotImplementedError
    
    def pick_action(self, observation):
        """Picks an action based on the policy and observation"""
        raise NotImplementedError