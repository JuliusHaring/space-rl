import sys
import json
from itertools import product
from matplotlib import cm

import numpy as np
import matplotlib.pyplot as plt
from agents.base import RLAgent
from environment.base import Environment
from environment.types import actions_dict, states_dict, action_space


class QLearningAgent(RLAgent):
    def __init__(
        self,
        env: Environment,
        alpha: float = 0.1,
        gamma: float = 0.99,
        epsilon: float = 0.1,
    ) -> None:
        super().__init__(env)
        self.epsilon = epsilon  # exploration rate
        self.alpha = alpha  # learning rate
        self.gamma = gamma  # discount factor
        self.q_table = {}

    def state_to_index(self, state: states_dict) -> int:
        """
        Converts a state dictionary into a unique index.

        This is a simplistic implementation assuming each state variable
        can take on a known fixed number of discrete values.
        """
        # Flatten the state into a list of values
        # flattened_state = []
        # for key in sorted(state.keys()):
        #     flattened_state.extend(state[key][0])  # Position tuple
        #     flattened_state.extend(state[key][1])  # Velocity tuple

        # # Convert list of state values into a string to use as a dictionary key
        # state_str = "_".join(map(str, flattened_state))

        state_str = json.dumps(state)

        # Ensure there is an entry for this state string in the Q-table
        if state_str not in self.q_table:
            self.q_table[state_str] = [-sys.maxsize] * len(self.get_action_space())

        return state_str

    def action_to_index(self, action: actions_dict) -> int:
        """
        Converts an action dictionary into a unique index.

        This is a simplistic implementation assuming each action variable
        can take on a known fixed number of discrete values.
        """
        # Flatten the action into a list of values
        flattened_action = []
        for key in sorted(action.keys()):
            flattened_action.extend(action[key])

        # Convert the action to its index in the action space
        try:
            action_index = self.get_action_space().index(tuple(flattened_action))
        except ValueError:
            # This would occur if an action is taken
            # that's not in the predefined action space
            raise ValueError("Action provided is not in the action space.")

        return action_index

    def get_action_space(self) -> action_space:
        return list(product(range(-1, 2), repeat=3))

    def select_action(self, state: states_dict) -> actions_dict:
        state_index = self.state_to_index(state)
        if np.random.rand() < self.epsilon:
            # Explore: select a random action
            action = self.get_action_space()[
                np.random.randint(0, len(self.get_action_space()))
            ]
        else:
            # Exploit: select the action with the highest Q-value for the current state
            action_index = np.argmax(self.q_table[state_index])  # This returns an index
            action = self.get_action_space()[action_index]  # Convert index to action
        # Convert the action tuple back into the actions_dict format
        action_dict = {list(state.keys())[0]: action}
        return action_dict

    def update_knowledge(
        self,
        state: states_dict,
        action: actions_dict,
        reward: float,
        next_state: states_dict,
        done: bool,
    ):
        state_index = self.state_to_index(state)
        next_state_index = self.state_to_index(next_state)
        action_index = self.action_to_index(action)

        # Q-Learning update
        if not done:
            max_future_q = np.max(self.q_table[next_state_index])
            current_q = self.q_table[state_index][action_index]
            new_q = (1 - self.alpha) * current_q + self.alpha * (
                reward + self.gamma * max_future_q
            )
        else:
            new_q = reward  # No future reward if the episode is done

        self.q_table[state_index][action_index] = new_q

    def train(self, total_episodes: int = 100, batch_size: int = 32):
        batch = []

        for episode in range(total_episodes):
            state = self.env.reset()
            total_reward = 0
            done = False

            while not done:
                action = self.select_action(state)
                next_state, reward, done, _ = self.env.step(action)

                batch.append((state, action, reward, next_state, done))

                if len(batch) == batch_size:
                    self.learn_from_batch(batch)
                    batch = []  # Clear the batch

                state = next_state
                total_reward += reward

            self.evaluate_best_env(reward=reward, environment=self.env)
            print(f"Episode: {episode}, Total reward: {total_reward}")
        self.render_best()
        self.render_q_table()

    def render_q_table(self):
        fig = plt.figure(figsize=(8, 6))
        ax = fig.add_subplot(111, projection="3d")

        # Create a list to collect all Q-values for normalization
        all_q_values = [
            action for actions in self.q_table.values() for action in actions
        ]

        # Normalize the Q-values between 0 and 1
        max_q_value = max(all_q_values)
        min_q_value = min(all_q_values)
        norm = plt.Normalize(min_q_value, max_q_value)

        # Create a scalar mappable for the color bar with the normalization
        scalar_map = cm.ScalarMappable(norm=norm, cmap="jet")

        agent_name = next(obj for obj in self.env.objects if obj.is_movable).name

        for state_str, actions in self.q_table.items():
            state = json.loads(state_str)[agent_name]
            max_q_value = max(actions)

            # Get the corresponding color for this Q-value
            color = scalar_map.to_rgba(max_q_value)

            # Plot the state with its corresponding Q-value color
            ax.scatter(state[0][0], state[0][1], state[0][2], color=color, marker="o")

        # Set labels for axes
        ax.set_xlabel("State Dimension 1")
        ax.set_ylabel("State Dimension 2")
        ax.set_zlabel("State Dimension 3")

        # Add the color bar using the scalar mappable
        cbar = fig.colorbar(scalar_map, shrink=0.5, aspect=5, ax=ax)
        cbar.set_label("Q-value")

        plt.show()

    def learn_from_batch(self, batch):
        for state, action, reward, next_state, done in batch:
            self.update_knowledge(state, action, reward, next_state, done)


if __name__ == "__main__":
    from environment.models import MovableObject
    from environment.p2p_environment import P2PEnvironment

    obj = MovableObject(name="Object", pos_x=0.0, pos_y=0.0, pos_z=0.0)
    env = P2PEnvironment(objects=[obj], size=10)

    ql = QLearningAgent(env=env)
    ql.train(total_episodes=100)
