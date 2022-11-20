import dataclasses
from typing import Any, Generic, TypeVar
from abc import ABC, abstractmethod
import copy
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from objects.field import Point2D, Object

VelType = TypeVar("VelType")


@dataclasses.dataclass(frozen=True)
class RobotState(Generic[VelType]):
    pos: Point2D
    vel: VelType


class RobotModel(ABC):
    _objects: list[Object]

    def __init__(self, objects: list[Object]):
        self._objects = objects

    def check_collision(self, state: RobotState, obstacles: list[Object]) -> bool:
        tmp_objects = self.get_objects(state)
        for obstacle in obstacles:
            for obj in tmp_objects:
                if obj.check_collision(obstacle):
                    return True
        return False

    def get_objects(self, state: RobotState) -> list[Object]:
        def calc_pos(pos_: Point2D, obj_: Object) -> Point2D:
            new_pos = obj_.pos.rotate(pos_.theta) + Point2D(pos_.x, pos_.y, 0.0)
            return new_pos

        def generate_new_obj(obj: Object, pos: Point2D) -> Object:
            ans = copy.deepcopy(obj)
            ans.change_pos(pos+ans.pos)
            return ans

        return list(map(lambda x: generate_new_obj(x, state.pos), self._objects))

    def plot(self, ax):
        for tmp in self._objects:
            tmp.plot(ax)
        ax.set_aspect("equal")


ActType = TypeVar("ActType")


class RobotModel_with_Dynamics(Generic[ActType], RobotModel):
    def __init__(self, objects: list[Object]):
        super(RobotModel_with_Dynamics, self).__init__(objects)

    @abstractmethod
    def step(self, state: RobotState, act: ActType, dt: float) -> RobotState:
        raise NotImplementedError()

    @abstractmethod
    def generate_next_act(self, state: RobotState, act_pre: ActType, config: Any) -> ActType:
        pass
