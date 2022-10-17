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
from typing import TYPE_CHECKING

import os

os.environ["NUMPY_EXPERIMENTAL_ARRAY_FUNCTION"] = "0"

import numpy as np
import numba as nb


@nb.jit(nopython=True, cache=True)
def ek(rho, rhou, rhov):
    _ek = 0.5 * (rhou**2 + rhov**2) / rho
    return _ek


@nb.jit(nopython=True, cache=True)
def Ek(rho, rhou, rhov):
    _ek = ek(rho, rhou, rhov)
    return _ek / rho


@nb.jit(nopython=True, cache=True)
def h(g, rho, rhou, rhov, e):
    _ek = ek(rho, rhou, rhov)
    return g * (e - _ek) + _ek


@nb.jit(nopython=True, cache=True)
def H(g, rho, rhou, rhov, e):
    _h = h(g, rho, rhou, rhov, e)
    return _h / rho


@nb.jit(nopython=True, cache=True)
def p(g, rho, rhou, rhov, e):
    _ek = ek(rho, rhou, rhov)
    return (g - 1) * (e - _ek)


@nb.jit(nopython=True, cache=True)
def a(g, rho, rhou, rhov, e):
    _p = p(g, rho, rhou, rhov, e)
    return np.sqrt(g * _p / rho)
