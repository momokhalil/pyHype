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

import numpy as np
from abc import abstractmethod
from pyHype.states.states import PrimitiveState, RoePrimitiveState


class FluxFunction:
    def __init__(self, inputs):
        self.inputs = inputs
        self.g = inputs.gamma
        self.nx = inputs.nx
        self.ny = inputs.ny
        self.n = inputs.n


    @staticmethod
    def wavespeeds_x(W: PrimitiveState):
        return W.u - W.a(), W.u + W.a()


    def harten_correction_x(self, Wroe: RoePrimitiveState, WL: PrimitiveState, WR: PrimitiveState):
        lambda_R_plus, lambda_R_minus = self.wavespeeds_x(WR)
        lambda_L_plus, lambda_L_minus = self.wavespeeds_x(WL)

        theta_1 = 2 * (lambda_R_plus - lambda_L_plus)
        theta_3 = 2 * (lambda_R_minus - lambda_L_minus)

        theta_1 = theta_1 * (theta_1 > 0)
        theta_3 = theta_3 * (theta_3 > 0)

        lambda_roe_minus = Wroe.v - Wroe.a()
        lambda_roe_plus = Wroe.v + Wroe.a()

        idx1 = np.absolute(lambda_roe_minus) < theta_1
        idx3 = np.absolute(lambda_roe_plus) < theta_3

        lambda_roe_minus[idx1] = 0.5 * (np.square(lambda_roe_minus[idx1]) / theta_1[idx1] + theta_1[idx1])
        lambda_roe_plus[idx3] = 0.5 * (np.square(lambda_roe_plus[idx3]) / theta_3[idx3] + theta_3[idx3])

        return lambda_roe_minus, lambda_roe_plus


    @staticmethod
    def wavespeeds_y(W: PrimitiveState):
        return W.v - W.a(), W.v + W.a()


    def harten_correction_y(self, Wroe: RoePrimitiveState, WL: PrimitiveState, WR: PrimitiveState):
        lambda_R_plus, lambda_R_minus = self.wavespeeds_y(WR)
        lambda_L_plus, lambda_L_minus = self.wavespeeds_y(WL)

        theta_1 = 2 * (lambda_R_plus - lambda_L_plus)
        theta_3 = 2 * (lambda_R_minus - lambda_L_minus)

        theta_1 = theta_1 * (theta_1 > 0)
        theta_3 = theta_3 * (theta_3 > 0)

        lambda_roe_minus = Wroe.v - Wroe.a()
        lambda_roe_plus = Wroe.v + Wroe.a()

        idx1 = np.absolute(lambda_roe_minus) < theta_1
        idx3 = np.absolute(lambda_roe_plus) < theta_3

        lambda_roe_minus[idx1] = 0.5 * (np.square(lambda_roe_minus[idx1]) / theta_1[idx1] + theta_1[idx1])
        lambda_roe_plus[idx3] = 0.5 * (np.square(lambda_roe_plus[idx3]) / theta_3[idx3] + theta_3[idx3])

        return lambda_roe_minus, lambda_roe_plus

    @abstractmethod
    def get_flux(self, UL, UR):
        pass
