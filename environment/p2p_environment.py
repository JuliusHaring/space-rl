from environment.base import Environment
from environment.models import MovableObject
from environment.types import float3d, episode_state
import numpy as np

import random


class P2PEnvironment(Environment):
    def __init__(
        self, objects: list[MovableObject], dt: float = 1.0, size: int = 100
    ) -> None:
        self.size = size
        self.target_distance = 1e-3
        t_x, t_y, t_z = self._sample_point()
        target = MovableObject(
            name="Target", pos_x=t_x, pos_y=t_y, pos_z=t_z, is_movable=False
        )
        super().__init__(objects + [target], dt)

    def _sample_point(self) -> float3d:
        return (
            random.uniform(-self.size, self.size),
            random.uniform(-self.size, self.size),
            random.uniform(-self.size, self.size),
        )

    def episode_finished(self) -> episode_state:
        target = next(obj for obj in self.objects if not obj.is_movable)
        agent = next(obj for obj in self.objects if obj.is_movable)

        states = self.get_states()
        agent_state = states[agent.name]
        target_state = states[target.name]

        if any(np.abs(p) >= self.size for p in agent_state[0]):
            return True, False

        if all(
            np.abs(p - t) < self.target_distance
            for p, t in zip(agent_state[0], target_state[0])
        ):
            return True, True

        return False, False

    def _calculate_reward(self) -> float:
        target = next(obj for obj in self.objects if not obj.is_movable)
        agent = next(obj for obj in self.objects if obj.is_movable)

        states = self.get_states()
        agent_state = states[agent.name]
        target_state = states[target.name]
        return -np.linalg.norm(np.array(agent_state[0]) - np.array(target_state[0]))


if __name__ == "__main__":
    obj = MovableObject(name="Object", pos_x=0.0, pos_y=0.0, pos_z=0.0)
    env = P2PEnvironment(objects=[obj])

    for i in range(env.size):
        env.step({obj.name: (1, 1, 1)})
        reward = env._calculate_reward()
        if env.episode_finished()[0]:
            break
    env.render()
