import logging
import numpy as np
from typing import Dict, List, Tuple
from reward_system.config import Config
from reward_system.exceptions import RewardSystemError
from reward_system.models import RewardModel
from reward_system.utils import calculate_velocity, calculate_flow

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class RewardSystem:
    """
    Reward calculation and shaping system.

    This class is responsible for calculating rewards based on the agent's actions and the environment's state.
    It uses the velocity-threshold and Flow Theory algorithms from the research paper to calculate rewards.
    """

    def __init__(self, config: Config):
        """
        Initialize the reward system.

        Args:
            config: Configuration object containing settings for the reward system.
        """
        self.config = config
        self.reward_model = RewardModel(config)

    def calculate_reward(self, state: Dict, action: Dict, next_state: Dict) -> float:
        """
        Calculate the reward for the given state, action, and next state.

        Args:
            state: Current state of the environment.
            action: Action taken by the agent.
            next_state: Next state of the environment.

        Returns:
            The calculated reward.
        """
        try:
            # Calculate velocity
            velocity = calculate_velocity(state, action, next_state)

            # Calculate flow
            flow = calculate_flow(state, action, next_state)

            # Calculate reward using velocity-threshold algorithm
            reward = self.velocity_threshold_reward(velocity, flow)

            # Calculate reward using Flow Theory algorithm
            reward += self.flow_theory_reward(flow)

            return reward

        except RewardSystemError as e:
            logger.error(f"Error calculating reward: {e}")
            return 0.0

    def velocity_threshold_reward(self, velocity: float, flow: float) -> float:
        """
        Calculate the reward using the velocity-threshold algorithm.

        Args:
            velocity: Velocity of the agent.
            flow: Flow of the environment.

        Returns:
            The calculated reward.
        """
        if velocity < self.config.velocity_threshold:
            return 0.0
        else:
            return flow * self.config.velocity_reward_multiplier

    def flow_theory_reward(self, flow: float) -> float:
        """
        Calculate the reward using the Flow Theory algorithm.

        Args:
            flow: Flow of the environment.

        Returns:
            The calculated reward.
        """
        return flow * self.config.flow_reward_multiplier

class RewardModel:
    """
    Reward model.

    This class is responsible for storing and retrieving reward settings.
    """

    def __init__(self, config: Config):
        """
        Initialize the reward model.

        Args:
            config: Configuration object containing settings for the reward model.
        """
        self.config = config

class Config:
    """
    Configuration object.

    This class is responsible for storing and retrieving configuration settings.
    """

    def __init__(self):
        """
        Initialize the configuration object.
        """
        self.velocity_threshold = 0.5
        self.velocity_reward_multiplier = 1.0
        self.flow_reward_multiplier = 1.0

class RewardSystemError(Exception):
    """
    Reward system error.

    This exception is raised when an error occurs in the reward system.
    """

    def __init__(self, message: str):
        """
        Initialize the reward system error.

        Args:
            message: Error message.
        """
        self.message = message

def calculate_velocity(state: Dict, action: Dict, next_state: Dict) -> float:
    """
    Calculate the velocity of the agent.

    Args:
        state: Current state of the environment.
        action: Action taken by the agent.
        next_state: Next state of the environment.

    Returns:
        The calculated velocity.
    """
    # Calculate velocity using the formula from the research paper
    velocity = np.sqrt((next_state["x"] - state["x"]) ** 2 + (next_state["y"] - state["y"]) ** 2) / (action["time"] - state["time"])
    return velocity

def calculate_flow(state: Dict, action: Dict, next_state: Dict) -> float:
    """
    Calculate the flow of the environment.

    Args:
        state: Current state of the environment.
        action: Action taken by the agent.
        next_state: Next state of the environment.

    Returns:
        The calculated flow.
    """
    # Calculate flow using the formula from the research paper
    flow = (next_state["flow"] - state["flow"]) / (action["time"] - state["time"])
    return flow