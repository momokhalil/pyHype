from __future__ import annotations

from typing import TYPE_CHECKING
from abc import ABC, abstractmethod

import numpy as np

import pyHype.states as states

if TYPE_CHECKING:
    from pyHype.states.base import State


class StateConverter(ABC):
    def convert(self, state: State):
        if isinstance(state, states.PrimitiveState):
            return self._from_primitive(state)
        if isinstance(state, states.ConservativeState):
            return self._from_conservative(state)
        raise TypeError("State has a type which is not defined in the converter.")

    @abstractmethod
    def _from_primitive(self, state: states.PrimitiveState) -> np.ndarray:
        raise NotImplementedError

    @abstractmethod
    def _from_conservative(self, state: states.ConservativeState) -> np.ndarray:
        raise NotImplementedError


class ConservativeConverter(StateConverter):
    def _from_primitive(self, state: states.PrimitiveState) -> np.ndarray:
        return np.dstack(
            (
                state.rho.copy(),
                state.rho * state.u,
                state.rho * state.v,
                state.p / (state.g - 1) + state.ek(),
            )
        )

    def _from_conservative(self, state: states.ConservativeState) -> np.ndarray:
        return state.Q.copy()


class PrimitiveConverter(StateConverter):
    def _from_primitive(self, state: states.PrimitiveState):
        return state.Q.copy()

    def _from_conservative(self, state: states.ConservativeState) -> np.ndarray:
        return np.dstack(
            (
                state.rho.copy(),
                state.u.copy(),
                state.v.copy(),
                (state.g - 1) * (state.e - state.ek()),
            )
        )
