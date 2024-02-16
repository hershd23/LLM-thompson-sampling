import numpy as np

def random_argmax(vector):
  """Helper function to select argmax at random... not just first one."""
  index = np.random.choice(np.where(vector == vector.max())[0])
  return index

class BaseBandit(object):
    """
    Base class for all the bandit algorithms
    """
    def __init__(self):
        """Initializes the bandit algorithm"""
        raise NotImplementedError

    def update_observation(self, arm, reward):
        """Updates the observation for the bandit algorithm"""
        raise NotImplementedError
    
    def pick_action(self, observation):
        """Picks an action based on the policy and observation"""
        raise NotImplementedError