from abc import ABC, abstractmethod
from copy import deepcopy
import sys

from environment.base import Environment
from environment.types import actions_dict, states_dict, action_space


class RLAgent(ABC):
    def __init__(self, env: Environment) -> None:
        self.env = env

        self.best_env_reward = -sys.maxsize
        self.best_env = None

    def evaluate_best_env(self, reward: float, environment: Environment):
        if reward > self.best_env_reward:
            self.best_env_reward = reward
            self.best_env = deepcopy(environment)

    def render_best(self):
        print("Best reward: " + str(self.best_env_reward))
        self.best_env.render()

    @abstractmethod
    def get_action_space(self) -> action_space:
        """
        Give back a list of all possible actions to take.

        Returns:
        - A list of possible actions.
        """
        ...

    @abstractmethod
    def select_action(self, state: states_dict) -> actions_dict:
        """
        Given the current state of the environment, select an action.

        Parameters:
        - state: The current state of the environment.

        Returns:
        - An action to be taken given the current state.
        """
        pass

    @abstractmethod
    def update_knowledge(
        self,
        state: states_dict,
        action: actions_dict,
        reward: float,
        next_state: states_dict,
        done: bool,
    ):
        """
        Update the agent's knowledge based on the experience.

        Parameters:
        - state: The current state from which the action was taken.
        - action: The action that was taken.
        - reward: The reward received after taking the action.
        - next_state: The next state the environment transitioned to after the action.
        - done: A boolean flag indicating if the episode has ended.
        """
        pass

    @abstractmethod
    def learn_from_batch(self):
        """
        Perform learning/updating the agent's knowledge from a batch of experiences.
        """
        pass
