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
import os
os.environ['NUMPY_EXPERIMENTAL_ARRAY_FUNCTION'] = '0'

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
        lambda_R_p, lambda_R_m = self.wavespeeds_x(WR)
        lambda_L_p, lambda_L_m = self.wavespeeds_x(WL)
        lambda_roe_p, lambda_roe_m = self.wavespeeds_x(Wroe)

        theta_p = 2 * (lambda_R_p - lambda_L_p)
        theta_m = 2 * (lambda_R_m - lambda_L_m)

        theta_p = theta_p * (theta_p > 0)
        theta_m = theta_m * (theta_m > 0)

        idx1 = np.absolute(lambda_roe_p) < theta_p
        idx3 = np.absolute(lambda_roe_m) < theta_m

        lambda_roe_p[idx1] = 0.5 * (np.square(lambda_roe_p[idx1]) / theta_p[idx1] + theta_p[idx1])
        lambda_roe_m[idx3] = 0.5 * (np.square(lambda_roe_m[idx3]) / theta_m[idx3] + theta_m[idx3])

        """lambda_roe_p = np.where(np.absolute(lambda_roe_p) < theta_p,
                                0.5 * ((lambda_roe_p ** 2) / (theta_p + 1e-8) + theta_p),
                                lambda_roe_p)

        lambda_roe_m = np.where(np.absolute(lambda_roe_m) < theta_m,
                                0.5 * ((lambda_roe_m ** 2) / (theta_m + 1e-8) + theta_m),
                                lambda_roe_m)"""

        return lambda_roe_p, lambda_roe_m


    @staticmethod
    def wavespeeds_y(W: PrimitiveState):
        return W.v - W.a(), W.v + W.a()


    def harten_correction_y(self, Wroe: RoePrimitiveState, WL: PrimitiveState, WR: PrimitiveState):
        lambda_R_p, lambda_R_m = self.wavespeeds_y(WR)
        lambda_L_p, lambda_L_m = self.wavespeeds_y(WL)
        lambda_roe_p, lambda_roe_m = self.wavespeeds_y(Wroe)

        theta_p = 2 * (lambda_R_p - lambda_L_p)
        theta_m = 2 * (lambda_R_m - lambda_L_m)

        theta_p = theta_p * (theta_p > 0)
        theta_m = theta_m * (theta_m > 0)

        idxp = np.absolute(lambda_roe_p) < theta_p
        idxm = np.absolute(lambda_roe_m) < theta_m

        lambda_roe_p[idxp] = 0.5 * (np.square(lambda_roe_p[idxp]) / theta_p[idxp] + theta_p[idxp])
        lambda_roe_m[idxm] = 0.5 * (np.square(lambda_roe_m[idxm]) / theta_m[idxm] + theta_m[idxm])

        """lambda_roe_p = np.where(np.absolute(lambda_roe_p) < theta_p,
                                0.5*((lambda_roe_p ** 2) / (theta_p + 1e-8) + theta_p),
                                lambda_roe_p)

        lambda_roe_m = np.where(np.absolute(lambda_roe_m) < theta_m,
                                0.5 * ((lambda_roe_m ** 2) / (theta_m + 1e-8) + theta_m),
                                lambda_roe_m)"""

        return lambda_roe_p, lambda_roe_m

    @abstractmethod
    def get_flux(self, UL, UR):
        pass
