from pydantic import BaseModel, PrivateAttr

from environment.types import float3d


class MovableObject(BaseModel):
    is_movable: bool = True
    name: str
    pos_x: float
    pos_y: float
    pos_z: float
    vel_x: float = 0
    vel_y: float = 0
    vel_z: float = 0
    _initial_state: tuple = PrivateAttr()

    def __init__(self, **data):
        super().__init__(**data)
        self.__dict__["_initial_state"] = self._get_state()

    def _get_state(self) -> tuple[float3d, float3d]:
        return (self.pos_x, self.pos_y, self.pos_z), (
            self.vel_x,
            self.vel_y,
            self.vel_z,
        )

    def _update_state(self, action: float3d, time_step: float) -> None:
        if self.is_movable:
            acc_x, acc_y, acc_z = action
            self.vel_x += acc_x * time_step
            self.vel_y += acc_y * time_step
            self.vel_z += acc_z * time_step
            self.pos_x += self.vel_x * time_step
            self.pos_y += self.vel_y * time_step
            self.pos_z += self.vel_z * time_step

    def reset(self):
        (
            (self.pos_x, self.pos_y, self.pos_z),
            (self.vel_x, self.vel_y, self.vel_z),
        ) = self._initial_state
