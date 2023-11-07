from abc import ABC, abstractmethod
from matplotlib.axes import Axes

import matplotlib.pyplot as plt
import numpy as np

from environment.models import MovableObject
from environment.types import states_dict, actions_dict, episode_state


class Environment(ABC):
    def __init__(self, objects: list[MovableObject], dt: float = 1.0) -> None:
        self.objects = objects
        self.t = 0
        self.dt = dt
        self.trajectories = [[] for _ in objects]

    @abstractmethod
    def episode_finished(self) -> episode_state:
        ...

    @abstractmethod
    def _calculate_reward(self) -> float:
        ...

    def reset(self):
        states = []
        for i, obj in enumerate(self.objects):
            obj.reset()
            self.trajectories[i] = []
            states.append(obj.get_state())
        self.t = 0
        return states

    def _get_states(self) -> states_dict:
        return {obj.name: obj.get_state() for obj in self.objects}

    @abstractmethod
    def _add_visuals(self, ax: Axes):
        ...

    def render(self):
        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")

        for obj, trajectory in zip(self.objects, self.trajectories):
            trajectory = np.array(
                trajectory
            )  # Convert trajectory to NumPy array for easy slicing
            ax.plot(
                trajectory[:, 0],
                trajectory[:, 1],
                trajectory[:, 2],
                label=f"Trajectory of object {obj.name}",
            )
            # Plot the current position with a different marker
            ax.scatter(
                *trajectory[-1],
                label=f"Current position of object {obj.name}",
                marker="o",
            )

        self._add_visuals(ax=ax)

        ax.set_xlabel("X Coordinate")
        ax.set_ylabel("Y Coordinate")
        ax.set_zlabel("Z Coordinate")
        ax.set_title("3D Projection of Objects")

        ax.grid(True)

        ax.set_xticks(range(-self.size, self.size, self.size // 10))
        ax.set_yticks(range(-self.size, self.size, self.size // 10))
        ax.set_zticks(range(-self.size, self.size, self.size // 10))

        ax.legend()

        plt.show()

    def step(self, actions: actions_dict) -> tuple[states_dict, float, bool, dict]:
        info = {}
        for i, obj in enumerate(self.objects):
            if obj.is_movable:  # Check if the object is movable and thus controllable
                action = actions.get(obj.name)
                obj._update_state(action=action, time_step=self.dt)
            # Append both position and velocity to the trajectories and next_states
            state = obj.get_state()
            self.trajectories[i].append(state[0])
        self.t += self.dt
        done, success = self.episode_finished()
        reward = self._calculate_reward()  # if success else 0
        next_states = self._get_states()
        return next_states, reward, done, info
