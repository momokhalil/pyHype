from __future__ import annotations

from typing import TYPE_CHECKING, Type, Callable
from abc import ABC, abstractmethod

import numpy as np

import pyhype.states as states

if TYPE_CHECKING:
    from pyhype.states.base import State
    from pyhype.solvers.base import ProblemInput


class StateConverter(ABC):

    @staticmethod
    def from_state(state: states.State, from_state: states.State) -> None:
        converter = from_state.get_class_type_converter()
        state.Q = converter.to(state_type=type(state))(from_state)

    @staticmethod
    def to_type(
        state: states.State,
        to_type: Type[states.State],
        inputs: ProblemInput,
    ) -> states.State:
        converter = state.get_class_type_converter()
        array = converter.to(to_type)(state)
        created = to_type(inputs=inputs, array=array)
        return created

class BaseConverter(ABC):
    """
    Defines interface for implementing State converters.
    """

    @classmethod
    def to(cls, state_type: Type[states.State]) -> Callable[[states.State], np.ndarray]:
        """
        Returns the conversion function that converts a Base type
        to a state_type.

        :param state_type:
        :return:
        """
        if state_type == states.PrimitiveState:
            return cls.to_primitive
        if state_type == states.ConservativeState:
            return cls.to_conservative

    @staticmethod
    @abstractmethod
    def to_primitive(state: states.State) -> np.ndarray:
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def to_conservative(state: states.State) -> np.ndarray:
        raise NotImplementedError


class ConservativeConverter(BaseConverter):
    """
    Converts `ConservativeState` objects to other `State` types.
    """

    @staticmethod
    def to_primitive(state: states.ConservativeState) -> np.ndarray:
        return np.dstack(
            (
                state.rho.copy(),
                state.u.copy(),
                state.v.copy(),
                (state.g - 1) * (state.e - state.ek()),
            )
        )

    @staticmethod
    def to_conservative(state: states.ConservativeState) -> np.ndarray:
        return state.Q.copy()


class PrimitiveConverter(BaseConverter):
    @staticmethod
    def to_primitive(state: states.PrimitiveState):
        return state.Q.copy()

    @staticmethod
    def to_conservative(state: states.PrimitiveState) -> np.ndarray:
        return np.dstack(
            (
                state.rho.copy(),
                state.rho * state.u,
                state.rho * state.v,
                state.p / (state.g - 1) + state.ek(),
            )
        )
