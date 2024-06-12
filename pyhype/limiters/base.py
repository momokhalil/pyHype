"""
Copyright 2021 Mohamed Khalil

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""
from __future__ import annotations

import os

os.environ["NUMPY_EXPERIMENTAL_ARRAY_FUNCTION"] = "0"

import numpy as np
import numba as nb
from abc import abstractmethod

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pyhype.blocks.quad_block import BaseBlockGhost


class SlopeLimiter:
    def __init__(self, config):
        self.config = config
        self.phi = np.zeros((config.ny, config.nx, 4))

    def limit(
        self,
        parent_block: BaseBlockGhost,
        gqpE: list[np.ndarray],
        gqpW: list[np.ndarray],
        gqpN: list[np.ndarray],
        gqpS: list[np.ndarray],
    ) -> None:
        self._limit(parent_block, gqpE, gqpW, gqpN, gqpS)

    def _get_slope(
        self,
        parent_block: BaseBlockGhost,
        gqpE: np.ndarray,
        gqpW: np.ndarray,
        gqpN: np.ndarray,
        gqpS: np.ndarray,
    ) -> [[np.ndarray]]:
        """
        Calculates the solution slopes to determine the slope limiter values on all quadrature points.

        :type parent_block: BaseBlockGhost
        :param parent_block: Reference block with solution data for slope calculation

        :type gqpE: np.ndarray
        :param gqpE: reconstructed solution states at east face quadrature points

        :type gqpW: np.ndarray
        :param gqpW: reconstructed solution states at west face quadrature points

        :type gqpN: np.ndarray
        :param gqpN: reconstructed solution states at north face quadrature points

        :type gqpS: np.ndarray
        :param gqpS: reconstructed solution states at south face quadrature points

        :rtype: tuple(list(np.ndarray))
        :return: lists containing the solution slopes at each quadraure point for each face
        """
        # Concatenate block solution state with ghost block solution states
        EW = np.concatenate(
            (
                parent_block.ghost.W.state.data,
                parent_block.state.data,
                parent_block.ghost.E.state.data,
            ),
            axis=1,
        )
        NS = np.concatenate(
            (
                parent_block.ghost.S.state.data,
                parent_block.state.data,
                parent_block.ghost.N.state.data,
            ),
            axis=0,
        )
        # Values for min/max evaluation
        vals = (parent_block.state.data, EW[:, :-2], EW[:, 2:], NS[:-2, :], NS[2:, :])
        # Difference between largest/smallest value and average value
        dmax = np.maximum.reduce(vals) - parent_block.state.data
        dmin = np.minimum.reduce(vals) - parent_block.state.data
        # Difference between quadrature points and average value
        dE = [_gqpE - parent_block.state.data for _gqpE in gqpE]
        dW = [_gqpW - parent_block.state.data for _gqpW in gqpW]
        dN = [_gqpN - parent_block.state.data for _gqpN in gqpN]
        dS = [_gqpS - parent_block.state.data for _gqpS in gqpS]
        # Calculate slopes for each face
        sE = [self._compute_slope(dmax, dmin, _dE.data) for _dE in dE]
        sW = [self._compute_slope(dmax, dmin, _dW.data) for _dW in dW]
        sN = [self._compute_slope(dmax, dmin, _dN.data) for _dN in dN]
        sS = [self._compute_slope(dmax, dmin, _dS.data) for _dS in dS]
        return sE, sW, sN, sS

    @staticmethod
    @nb.njit(cache=True)
    def _get_min_max(state, east_west, north_south):
        dmax = np.zeros_like(state)
        dmin = np.zeros_like(state)
        for i in range(state.shape[0]):
            for j in range(state.shape[1]):
                for k in range(state.shape[2]):
                    ew_minus = east_west[i, j, k]
                    ew_plus = east_west[i, j + 2, k]
                    ns_minus = north_south[i, j, k]
                    ns_plus = north_south[i + 2, j, k]
                    _state_val = state[i, j, k]
                    ew_max_min = (
                        (ew_minus, ew_plus)
                        if ew_minus > ew_plus
                        else (ew_plus, ew_minus)
                    )
                    ns_max_min = (
                        (ns_minus, ns_plus)
                        if ns_minus > ns_plus
                        else (ns_plus, ns_minus)
                    )
                    maximum = (
                        ew_max_min[0]
                        if ew_max_min[0] > ns_max_min[0]
                        else ns_max_min[0]
                    )
                    minimum = (
                        ew_max_min[1]
                        if ew_max_min[1] < ns_max_min[1]
                        else ns_max_min[1]
                    )
                    dmax[i, j, k] = maximum - _state_val
                    dmin[i, j, k] = minimum - _state_val
        return dmax, dmin

    def _limit(
        self,
        parent_block: BaseBlockGhost,
        gqpE: list[np.ndarray],
        gqpW: list[np.ndarray],
        gqpN: list[np.ndarray],
        gqpS: list[np.ndarray],
    ) -> None:
        """
        Calculates the solution slopes to determine the slope limiter values on all quadrature points.

        :type parent_block: BaseBlockGhost
        :param parent_block: Reference block with solution data for slope calculation

        :type gqpE: np.ndarray
        :param gqpE: reconstructed solution states at east face quadrature points

        :type gqpW: np.ndarray
        :param gqpW: reconstructed solution states at west face quadrature points

        :type gqpN: np.ndarray
        :param gqpN: reconstructed solution states at north face quadrature points

        :type gqpS: np.ndarray
        :param gqpS: reconstructed solution states at south face quadrature points

        :rtype: tuple(list(np.ndarray))
        :return: lists containing the solution slopes at each quadraure point for each face
        """
        # Calculate values needed to build slopes
        sE, sW, sN, sS = self._get_slope(parent_block, gqpE, gqpW, gqpN, gqpS)
        # Minimum limiter value
        phi = np.minimum.reduce(
            (
                *(self._limiter_func(_sE) for _sE in sE),
                *(self._limiter_func(_sW) for _sW in sW),
                *(self._limiter_func(_sN) for _sN in sN),
                *(self._limiter_func(_sS) for _sS in sS),
            )
        )
        self.phi = np.where(phi < 0, 0, phi)

    @staticmethod
    @nb.njit(cache=True)
    def _compute_slope(
        dmax: np.ndarray, dmin: np.ndarray, davg: np.ndarray
    ) -> np.ndarray:
        """
        Calculates the slope of the solution state based on the difference between the average solution state and
        the min/max average solution states in the reconstruction neighbots and the reconstructed solution states at
        at the quadrature points.

        :type dmax: np.ndarray
        :param dmax: Difference between the average soltion and the maximum average solution in all reconstruction
                     neighbors

        :type dmin: np.ndarray
        :param dmin: Difference between the average soltion and the minimum average solution in all reconstruction
                     neighbors

        :type davg: np.ndarray
        :param davg: Difference between the average solution and the reconstructed solutions at the quadrature points

        :rtype _s: np.ndarray
        :return _s: slope used for limiter calculation
        """
        s = np.ones_like(davg)
        for i in range(davg.shape[0]):
            for j in range(davg.shape[1]):
                for v in range(davg.shape[2]):
                    if davg[i, j, v] > 0:
                        s[i, j, v] = dmax[i, j, v] / davg[i, j, v]
                    elif davg[i, j, v] < 0:
                        s[i, j, v] = dmin[i, j, v] / davg[i, j, v]
        return s

    @staticmethod
    @abstractmethod
    def _limiter_func(slope: np.ndarray) -> np.ndarray:
        raise NotImplementedError
