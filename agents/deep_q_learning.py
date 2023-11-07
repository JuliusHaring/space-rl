import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import deque
import random
from environment.base import Environment
from environment.types import states_dict
from agents.q_learning import QLearningAgent


class DQN(nn.Module):
    def __init__(self, state_size, action_size):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(state_size, 24)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(24, 24)
        self.fc3 = nn.Linear(24, action_size)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        return self.fc3(x)


class DeepQLearningAgent(QLearningAgent):
    def __init__(
        self,
        env: Environment,
        state_size: int,
        action_size: int,
        buffer_size: int = 1000,
        batch_size: int = 32,
        alpha: float = 0.001,
        gamma: float = 0.99,
        epsilon: float = 0.1,
    ):
        super().__init__(env, alpha, gamma, epsilon)
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=buffer_size)
        self.batch_size = batch_size
        self.model = DQN(state_size, action_size)
        self.optimizer = optim.Adam(self.model.parameters(), lr=alpha)
        self.epsilon = epsilon

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def state_dict_to_network_input(self, state_dict: states_dict):
        state_values = []
        for key in sorted(state_dict.keys()):
            # Extract position and velocity (or other state variables) and flatten them
            state_values.extend(state_dict[key][0])  # Position tuple
            state_values.extend(state_dict[key][1])  # Velocity tuple
        # Convert the flattened list to a numpy array and reshape for the network
        return np.array(state_values, dtype=np.float32).reshape(
            1, -1
        )  # Reshape to 2D array with shape (1, num_state_variables)

    def select_action(self, state):
        state = torch.FloatTensor(state).unsqueeze(0)
        if np.random.rand() <= self.epsilon:
            return random.choice(np.arange(self.action_size))
        self.model.eval()
        with torch.no_grad():
            action_values = self.model(state)
        self.model.train()
        return np.argmax(action_values.cpu().data.numpy())

    def replay(self):
        if len(self.memory) < self.batch_size:
            return
        minibatch = random.sample(self.memory, self.batch_size)
        for state, action, reward, next_state, done in minibatch:
            state = torch.FloatTensor(state)
            next_state = torch.FloatTensor(next_state)
            action = torch.LongTensor([action])
            reward = torch.FloatTensor([reward])
            done = torch.FloatTensor([done])

            Q_targets_next = self.model(next_state).detach().max(1)[0].unsqueeze(1)
            Q_targets = reward + (self.gamma * Q_targets_next * (1 - done))
            Q_expected = self.model(state).gather(1, action.unsqueeze(1))

            loss = nn.MSELoss()(Q_expected, Q_targets)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        if self.epsilon > 0.01:
            self.epsilon *= 0.995

    def train(self, total_episodes=100):
        agent_name = next(obj for obj in self.env.objects if obj.is_movable).name
        for episode in range(total_episodes):
            state = self.env.reset()
            state = self.state_dict_to_network_input(state)
            total_reward = 0
            done = False
            while not done:
                action_idx = self.select_action(state)
                action = self.env.get_action_space()[action_idx]
                next_state, reward, done, _ = self.env.step({agent_name: action})
                next_state = self.state_dict_to_network_input(next_state)
                self.remember(state, action_idx, reward, next_state, done)
                state = next_state
                total_reward += reward
                self.replay()
            print(f"Episode: {episode}, Total reward: {total_reward}")


# Example usage
if __name__ == "__main__":
    from environment.p2p_environment import P2PEnvironment
    from environment.models import MovableObject

    obj = MovableObject(name="Object", pos_x=0.0, pos_y=0.0, pos_z=0.0)
    env = P2PEnvironment(objects=[obj], size=10)
    state_size = 12
    action_size = len(env.get_action_space())

    agent = DeepQLearningAgent(env, state_size, action_size)
    agent.train(total_episodes=100)
