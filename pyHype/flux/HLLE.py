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

import numba
import numpy as np
from pyHype.flux.base import FluxFunction
from pyHype.states.states import RoePrimitiveState, ConservativeState, PrimitiveState


class FluxHLLE(FluxFunction):
    def compute_flux(self,
                     WL: PrimitiveState,
                     WR: PrimitiveState
                     ) -> np.ndarray:

        # Get Roe state
        Wroe = RoePrimitiveState(self.inputs, WL, WR)

        # Left and Right wavespeeds
        L_p, L_m = self.wavespeeds_x(WL)
        R_p, R_m = self.wavespeeds_x(WR)

        # Harten entropy correction
        Lp, Lm = self.harten_correction_x(Wroe, WL, WR, L_p=L_p, L_m=L_m, R_p=R_p, R_m=R_m)

        L_plus = np.maximum.reduce((R_m, Lm))[:, :, None]
        L_minus = np.minimum.reduce((L_p, Lp))[:, :, None]

        UR = WR.to_conservative_state()
        UL = WL.to_conservative_state()

        FluxR = WR.F(U=UR)
        FluxL = WL.F(U=UL)
        Flux = (L_plus * FluxL - L_minus * FluxR + L_minus * L_plus * (UR - UL)) / (L_plus - L_minus)

        Flux = np.where(L_minus >= 0, FluxL, Flux)
        Flux = np.where(L_plus <= 0, FluxR, Flux)

        return Flux
