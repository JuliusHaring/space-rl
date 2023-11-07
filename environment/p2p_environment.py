from matplotlib.axes import Axes
from environment.base import Environment
from environment.models import MovableObject
from environment.types import float3d, episode_state
import numpy as np

import random


class P2PEnvironment(Environment):
    def __init__(self, objects: list[MovableObject], dt: float = 1.0) -> None:
        super().__init__(objects, dt)
        self.size = 1000
        self.target_distance = 1e-3
        self.targets = [self._sample_point() for _ in objects]

    def _add_visuals(self, ax: Axes):
        for obj, target in zip(self.objects, self.targets):
            ax.scatter(
                *target, label=f"Target position of object {obj.name}", marker="o"
            )

    def _sample_point(self) -> float3d:
        return (
            random.uniform(-self.size, self.size),
            random.uniform(-self.size, self.size),
            random.uniform(-self.size, self.size),
        )

    def episode_finished(self) -> episode_state:
        for obj, target in zip(self.objects, self.targets):
            (x, y, z), _ = obj.get_state()
            if any(np.abs(p) >= self.size for p in (x, y, z)):
                return True, False

            if all(
                np.abs(p - t) < self.target_distance for p, t in zip((x, y, z), target)
            ):
                return True, True

        return False, False

    def _calculate_reward(self) -> float:
        pos, _ = self.objects[0].get_state()
        target = self.targets[0]
        return -np.linalg.norm(np.array(pos) - np.array(target))


if __name__ == "__main__":
    obj = MovableObject(name="Object", pos_x=0.0, pos_y=0.0, pos_z=0.0)
    env = P2PEnvironment(objects=[obj])

    for i in range(env.size):
        env.step({obj.name: (1, 1, 1)})
        reward = env._calculate_reward()
        if env.episode_finished()[0]:
            break
    env.render()
