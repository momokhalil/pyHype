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

import numba as nb
import numpy as np
from pyhype.flux.base import FluxFunction
from pyhype.states.conservative import ConservativeState
from pyhype.states.primitive import PrimitiveState, RoePrimitiveState


class FluxHLLL(FluxFunction):
    def compute_flux(self, WL: PrimitiveState, WR: PrimitiveState) -> np.ndarray:

        # Get Roe state
        Wroe = RoePrimitiveState(self.inputs, WL, WR)
        # Left and Right wavespeeds
        L_p, L_m = self.wavespeeds_x(WL)
        R_p, R_m = self.wavespeeds_x(WR)
        # Harten entropy correction
        Lp, Lm = self.harten_correction_x(
            Wroe, WL, WR, L_p=L_p, L_m=L_m, R_p=R_p, R_m=R_m
        )
        L_plus = np.maximum.reduce((R_m, Lm))[:, :, None]
        L_minus = np.minimum.reduce((L_p, Lp))[:, :, None]
        # Left and right fluxes
        UR = ConservativeState(self.inputs, state=WR)
        UL = ConservativeState(self.inputs, state=WL)
        FluxR = WR.F(U=UR)
        FluxL = WL.F(U=UL)
        return self._HLLL_flux_JIT(
            Wroe.u, Wroe.a(), FluxL, FluxR, UL.U, UR.U, L_minus, L_plus
        )

    @staticmethod
    def _HLLL_flux_numpy(Wroe, FL, FR, UL, UR, L_minus, L_plus):

        # Get alhpa
        _u = Wroe.u[:, :, None]
        k = Wroe.a()[:, :, None] * (UR - UL)
        a = 1 - np.linalg.norm(FR - FR - k, axis=2) / (
            np.linalg.norm(k, axis=2) + 1e-14
        )
        a = np.where(a < 0, 0, a)[:, :, None]
        b = 1 - (1 - np.maximum(_u / L_plus, _u / L_minus)) * a

        # Compute flux
        Flux = (L_plus * FL - L_minus * FR + L_minus * L_plus * b * (UR - UL)) / (
            L_plus - L_minus
        )
        Flux = np.where(L_minus >= 0, FL, Flux)
        Flux = np.where(L_plus <= 0, FR, Flux)
        return Flux

    @staticmethod
    @nb.njit(cache=True)
    def _HLLL_flux_JIT(u_roe, a_roe, FL, FR, UL, UR, L_minus, L_plus):
        _flux = np.zeros_like(FL)
        for i in range(_flux.shape[0]):
            for j in range(_flux.shape[1]):
                _Lm = L_minus[i, j, 0]
                _Lp = L_plus[i, j, 0]
                if _Lm >= 0:
                    _flux[i, j, :] = FL[i, j, :]
                elif _Lp <= 0:
                    _flux[i, j, :] = FR[i, j, :]
                else:
                    u = u_roe[i, j]
                    dU = UR[i, j, :] - UL[i, j, :]
                    dF = FR[i, j, :] - FL[i, j, :]
                    k = a_roe[i, j] * np.linalg.norm(dU)

                    if k < 1e-16:
                        alpha = np.maximum(
                            0, 1 - np.linalg.norm(dF - u * dU) / (k + 1e-14)
                        )
                    else:
                        alpha = np.maximum(0, 1 - np.linalg.norm(dF - u * dU) / k)

                    _flux[i, j, :] = (
                        _Lp * FL[i, j, :]
                        - _Lm * FR[i, j, :]
                        + _Lm
                        * _Lp
                        * (1 - alpha * (1 - np.maximum(u / _Lm, u / _Lp)))
                        * dU
                    ) / (_Lp - _Lm)
        return _flux
