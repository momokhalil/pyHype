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
from pyHype.states.base import State
from pyHype.input.input_file_builder import ProblemInput


class PrimitiveState(State):
    """
    #Primitive Solution State#
    A class that represents the solution state vector of the 2D inviscid Euler equations in primitive form. The
    primitive solution vector, $W$, is represented mathematically as:
    $W = \\begin{bmatrix} \\rho \\ u \\ v \\ p \\end{bmatrix}^T$, where $\\rho$ is the solution density,
    $u$ is the velocity in the $x$-direction, $v$ is the velocity in the $y$-direction, and $p$ is the solution
    pressure. `PrimitiveState` can be used to represent the solution state of a QuadBlock in a solver that utilizes
    a primitive formulation. It can also be used to represent the solution state in BoundaryBlocks in a solver that
    utilizes a primitive formulation. Another primary use-case for `PrimitiveState` is converting a `ConservativeState`
    into `PrimitiveState` in order to access primitive solution variables if needed (e.g. flux functions).
    """

    def __init__(self,
                 inputs: ProblemInput,
                 nx: int = None,
                 ny: int = None,
                 state: State = None,
                 U_vector: np.ndarray = None,
                 W_vector: np.ndarray = None):
        """
        ## Attributes

        **Public**                                  \n
            W       Primitive state vector          \n
            rho     density                         \n
            u       x-direction velocity            \n
            v       y-direction velocity            \n
            p       pressure                        \n
        """

        # Initialize attributes for storing solution data

        # Check if an input vector is given
        if state or (U_vector is not None) or (W_vector is not None):
            _vec_given = True
        else:
            _vec_given = False

        # Check if more than one input vector is given
        if _vec_given and sum(map(bool, [state, U_vector is not None, W_vector is not None])) > 1:
            raise ValueError('Please provide only one type of input state or vector')

        # Get vector size
        _nx, _ny = 0, 0

        if ny and nx and not _vec_given:
            _nx, _ny = int(nx), int(ny)
        elif _vec_given:
            if state:
                _nx, _ny = state.nx, state.ny
            elif U_vector is not None:
                _nx, _ny = int(U_vector.shape[1]), int(U_vector.shape[0])
            elif W_vector is not None:
                _nx, _ny = int(W_vector.shape[1]), int(W_vector.shape[0])
        else:
            ValueError('Please either provide the size of the vector or a state class or a state vector')

        super().__init__(inputs, _nx, _ny)

        if state:
            if isinstance(state, PrimitiveState):
                self.from_primitive_state(state)
            elif isinstance(state, ConservativeState):
                self.from_conservative_state(state)

        elif U_vector is not None:
            self.from_conservative_state_vector(U_vector)

        elif W_vector is not None:
            self.from_conservative_state_vector(W_vector)

    # ------------------------------------------------------------------------------------------------------------------
    # Define primitive state properties

    # Density property (refers to q0 in Q)
    @property
    def rho(self) -> np.ndarray:
        return self.q0

    @rho.setter
    def rho(self, rho: np.ndarray) -> None:
        self.q0 = rho

    # u property (refers to q1 in Q)
    @property
    def u(self) -> np.ndarray:
        return self.q1

    @u.setter
    def u(self, u: np.ndarray) -> None:
        self.q1 = u

    # v property (refers to q2 in Q)
    @property
    def v(self) -> np.ndarray:
        return self.q2

    @v.setter
    def v(self, v: np.ndarray) -> None:
        self.q2 = v

    # p property (refers to q3 in Q)
    @property
    def p(self) -> np.ndarray:
        return self.q3

    @p.setter
    def p(self, p: np.ndarray) -> None:
        self.q3 = p

    # W property (refers to q3 in Q)
    @property
    def W(self) -> np.ndarray:
        return self.Q

    # W property (refers to q3 in Q)
    @W.setter
    def W(self, W: np.ndarray) -> None:
        self.Q = W

    # ------------------------------------------------------------------------------------------------------------------
    # METHODS FOR UPDATING INTERNAL STATE BASED ON EXTERNAL INPUTS

    def from_conservative_state(self, U: 'ConservativeState') -> None:
        """
        Creates a `PrimitiveState` object from a `ConservativeState` object. Given that `PrimitiveState` state
        vector is $W = \\begin{bmatrix} \\rho \\ u \\ v \\ p \\end{bmatrix}^T$ and `ConservativeState` state vector is
        $U = \\begin{bmatrix} \\rho \\ \\rho u \\ \\rho v \\ e \\end{bmatrix}^T$, and labelling the components of $W$ as
        $\\begin{bmatrix} w_1 \\ w_2 \\ w_3 \\ w_4 \\end{bmatrix}^T$ and the components of $U$ as
        $\\begin{bmatrix} u_1 \\ u_2 \\ u_3 \\ u_4 \\end{bmatrix}^T$, then, the transformation from $U$ to $W$ is: \n
        $\\begin{bmatrix} w_1 \\\\\ w_2 \\\\\ w_3 \\\\\ w_4 \\end{bmatrix} =
        \\begin{bmatrix} u_1 \\\\\ u_2 / u_1 \\\\\ u_3 / u_1 \\\\\ (\\gamma - 1)[u_4 - \\frac{1}{2 \\rho}(u_2^2 + u_3^2)] \\\\\ \\end{bmatrix} =
        \\begin{bmatrix} \\rho \\\\\ \\rho u / \\rho \\\\\ \\rho v / \\rho \\\\\ (\\gamma - 1)[e - \\frac{1}{2 \\rho}((\\rho u)^2 + (\\rho v)^2)] \\\\\ \\end{bmatrix} =
        \\begin{bmatrix} \\rho \\\\\ u \\\\\ v \\\\\ p \\end{bmatrix}$

        **Parameters**                              \n
            U       ConservativeState object        \n
        """
        self.rho = U.rho.copy()
        self.u = U.rhou / self.rho
        self.v = U.rhov / self.rho
        self.p = (self.g - 1) * (U.e - 0.5 * (U.rhou**2 + U.rhov**2) / self.rho)

        self.set_state_from_vars()

    def from_conservative_state_vector(self, U_vector: np.ndarray):
        """

        """
        self.rho = U_vector[:, :, 0].copy()
        self.u = U_vector[:, :, 1] / self.rho
        self.v = U_vector[:, :, 2] / self.rho
        self.p = (self.g - 1) * (U_vector[:, :, 3] - 0.5 * self.rho * (self.u ** 2 + self.v ** 2))

        self.set_state_from_vars()

    def from_conservative_state_vars(self, rho: np.ndarray, rhou: np.ndarray, rhov: np.ndarray, e: np.ndarray) -> None:
        """
        Populates the primitive state vector and variables W with their appropriate components based on conservative
        state variables

        **Parameters**                              \n
            rho     density                         \n
            u       x-direction velocity            \n
            v       y-direction velocity            \n
            p       pressure                        \n
        """

        # Set density, u, v and pressure
        self.rho = rho.copy()
        self.u = rhou / rho
        self.v = rhov / rho
        self.p = (self.g - 1) * (e - 0.5 * (rhou ** 2 + rhov ** 2) / rho)

        # Set W components appropriately
        self.set_state_from_vars()

    def from_primitive_state(self, W: 'PrimitiveState') -> None:
        """

        """
        self.W = W.W.copy()
        self.set_vars_from_state()

    def from_primitive_state_vector(self, W_vector: np.ndarray) -> None:
        """

        """
        self.W = W_vector.copy()
        self.set_vars_from_state()

    def from_primitive_state_vars(self, rho: np.ndarray, u: np.ndarray, v: np.ndarray, p: np.ndarray) -> None:
        """
        Populates the primitive state vector and variables W with their appropriate components based on primitive state
        variables

        **Parameters**                              \n
            rho     density                         \n
            u       x-direction velocity            \n
            v       y-direction velocity            \n
            p       pressure                        \n
        """

        # Set density, u, v and pressure
        self.rho = rho.copy()
        self.u = u.copy()
        self.v = v.copy()
        self.p = p.copy()

        # Set W components appropriately
        self.set_state_from_vars()

    def to_conservative_state(self) -> 'ConservativeState':
        """
        Creates a `ConservativeState` object from itself
        """
        return ConservativeState(self.inputs, W_vector=self.W)

    def H(self) -> np.ndarray:
        return (self.g / (self.g - 1)) * (self.p / self.rho) + np.mean(self.u**2 + self.v**2)

    def a(self) -> np.ndarray:
        return np.sqrt(self.g * self.p / self.rho)

    def non_dim(self) -> None:
        """
        Non-dimentionalizes the primitive state vector W. Let the non-dimentionalized primitive state vector be
        $W^* = \\begin{bmatrix} \\rho^* \\ u^* \\ v^* \\ p^* \\end{bmatrix}^T$, where $^*$ indicates a
        non-dimentionalized quantity. The non-dimentionalized primitive variables are:  \n
        $\\rho^* = \\rho/\\rho_\\infty$                                                 \n
        $u^* = u/a_\\infty$                                                             \n
        $u^* = v/a_\\infty$                                                             \n
        $p^* = p/\\rho_\\infty a_\\infty^2$                                             \n
        If `PrimitiveState` is created from a non-dimentionalized `ConservativeState`, it will be non-dimentional.
        """

        # Non-dimentionalize each component of W
        self.W[:, :, 0] /= self.inputs.rho_inf
        self.W[:, :, 1] /= self.inputs.rho_inf * self.inputs.a_inf
        self.W[:, :, 2] /= self.inputs.rho_inf * self.inputs.a_inf
        self.W[:, :, 3] /= self.inputs.rho_inf * self.inputs.a_inf ** 2

        # Set variables from non-dimensionalized W
        self.set_vars_from_state()


class ConservativeState(State):
    """
    #Conservative Solution State#
    A class that represents the solution state vector of the 2D inviscid Euler equations in conservative form. The
    conservative solution vector, $U$, is represented mathematically as:
    $U = \\begin{bmatrix} \\rho \\ \\rho u \\ \\rho v \\ e \\end{bmatrix}^T$, where $\\rho$ is the solution density,
    $u$ is the $x$-direction momentum per unit volume, $v$ is the $y$-direction momentum per unit volume, and $e$ is
    the energy per unit volume. `ConservativeState` can be used to represent the solution state of a QuadBlock in a
    solver that utilizes a conservative formulation. It can also be used to represent the solution state in
    BoundaryBlocks in a solver that utilizes a conservative formulation.
    """

    def __init__(self,
                 inputs: ProblemInput,
                 nx: int = None,
                 ny: int = None,
                 state: State = None,
                 U_vector: np.ndarray = None,
                 W_vector: np.ndarray = None):
        """
        ## Attributes

        **Public**                                                  \n
            U       Conservative state vector                       \n
            rho     density                                         \n
            rhou    x-direction momentum per unit volume            \n
            rhov    y-direction momentum per unit volume            \n
            e       energy per unit volume                          \n
        """

        # Check if an input vector is given
        if state or (U_vector is not None) or (W_vector is not None):
            _vec_given = True
        else:
            _vec_given = False

        # Check if more than one input vector is given
        if _vec_given and sum(map(bool, [state, U_vector is not None, W_vector is not None])) > 1:
            raise ValueError('Please provide only one type of input state or vector')

        # Get vector size
        _nx, _ny = 0, 0

        if ny and nx and not _vec_given:
            _nx, _ny = int(nx), int(ny)
        elif _vec_given:
            if state:
                _nx, _ny = state.nx, state.ny
            elif U_vector is not None:
                _nx, _ny = int(U_vector.shape[1]), int(U_vector.shape[0])
            elif W_vector is not None:
                _nx, _ny = int(W_vector.shape[1]), int(W_vector.shape[0])
        else:
            ValueError('Please either provide the size of the vector or a state class or a state vector')

        super().__init__(inputs, _nx, _ny)

        if state:
            if isinstance(state, PrimitiveState):
                self.from_primitive_state(state)
            elif isinstance(state, ConservativeState):
                self.from_conservative_state(state)

        elif U_vector is not None:
            self.from_conservative_state_vector(U_vector)

        elif W_vector is not None:
            self.from_conservative_state_vector(W_vector)

    # ------------------------------------------------------------------------------------------------------------------
    # Define primitive state properties

    # rho property (refers to q0 in Q)
    @property
    def rho(self):
        return self.q0

    @rho.setter
    def rho(self, rho: np.ndarray) -> None:
        self.q0 = rho

    # u property (refers to q1 in Q)
    @property
    def rhou(self):
        return self.q1

    @rhou.setter
    def rhou(self, rhou: np.ndarray) -> None:
        self.q1 = rhou

    # v property (refers to q2 in Q)
    @property
    def rhov(self):
        return self.q2

    @rhov.setter
    def rhov(self, rhov: np.ndarray) -> None:
        self.q2 = rhov

    # e property (refers to q3 in Q)
    @property
    def e(self):
        return self.q3

    @e.setter
    def e(self, e: np.ndarray) -> None:
        self.q3 = e

    # W property (refers to q3 in Q)
    @property
    def U(self):
        return self.Q

    # W property (refers to q3 in Q)
    @U.setter
    def U(self, U: np.ndarray):
        self.Q = U

    # PUBLIC METHODS ---------------------------------------------------------------------------------------------------

    def from_conservative_state(self, U: 'ConservativeState') -> None:
        self.U = U.U.copy()
        self.set_vars_from_state()

    def from_conservative_state_vector(self, U_vector: np.ndarray) -> None:
        self.U = U_vector.copy()
        self.set_vars_from_state()

    def from_conservative_state_vars(self,
                                     rho: np.ndarray,
                                     rhou: np.ndarray,
                                     rhov: np.ndarray,
                                     e: np.ndarray) -> None:
        """
        Populates the primitive state vector and variables W with their appropriate components

        **Parameters**                                              \n
            rho     density                                         \n
            rhou    x-direction momentum per unit volume            \n
            rhov    y-direction momentum per unit volume            \n
            e       energy per unit volume                          \n
        """

        # Set density, u, v and pressure
        self.rho = rho.copy()
        self.rhou = rhou.copy()
        self.rhov = rhov.copy()
        self.e = e.copy()

        # Set state vector from state variables
        self.set_state_from_vars()

    def from_primitive_state(self, W: PrimitiveState) -> None:
        """
        Creates a `COnservativeState` object from a `PrimitiveState` object. Given that `PrimitiveState` state
        vector is $W = \\begin{bmatrix} \\rho \\ u \\ v \\ p \\end{bmatrix}^T$ and `ConservativeState` state vector is
        $U = \\begin{bmatrix} \\rho \\ \\rho u \\ \\rho v \\ e \\end{bmatrix}^T$, and labelling the components of $W$ as
        $\\begin{bmatrix} w_1 \\ w_2 \\ w_3 \\ w_4 \\end{bmatrix}^T$ and the components of $U$ as
        $\\begin{bmatrix} u_1 \\ u_2 \\ u_3 \\ u_4 \\end{bmatrix}^T$, then, the transformation from $U$ to $W$ is: \n
        $\\begin{bmatrix} u_1 \\\\\ u_2 \\\\\ u_3 \\\\\ u_4 \\end{bmatrix} =
        \\begin{bmatrix} w_1 \\\\\ w_2w_1 \\\\\ w_3w_1 \\\\\ \\frac{w_4}{\\gamma - 1} + \\frac{w_1}{2}(w_2^2 + w_3^2) \\end{bmatrix} =
        \\begin{bmatrix} \\rho \\\\\ \\rho u \\\\\ \\rho v \\\\\ \\frac{p}{\\gamma - 1} + \\frac{\\rho}{2} (u^2 + v^2) \\end{bmatrix} =
        \\begin{bmatrix} \\rho \\\\\ \\rho u \\\\\ \\rho v \\\\\ e \\end{bmatrix}$

        **Parameters**                              \n
            W       PrimitiveState object           \n
        """
        self.rho    = W.rho.copy()
        self.rhou   = W.rho * W.u
        self.rhov   = W.rho * W.v
        self.e      = W.p / (self.g - 1) + 0.5 * W.rho * (W.u**2 + W.v**2)

        self.set_state_from_vars()

    def from_primitive_state_vector(self, W_vector: np.ndarray) -> None:
        """

        """
        self.rho    = W_vector[:, :, 0].copy()
        self.rhou   = self.rho * W_vector[:, :, 1]
        self.rhov   = self.rho * W_vector[:, :, 2]
        self.e      = 0.5 * (self.rhou ** 2 + self.rhov ** 2) / self.rho + W_vector[:, :, 3] / (self.g - 1)


        self.set_state_from_vars()



    def from_primitive_state_vars(self,
                                     rho: np.ndarray,
                                     u: np.ndarray,
                                     v: np.ndarray,
                                     p: np.ndarray) -> None:
        """
        Populates the primitive state vector and variables W with their appropriate components

        **Parameters**                                              \n
            rho     density                                         \n
            rhou    x-direction momentum per unit volume            \n
            rhov    y-direction momentum per unit volume            \n
            e       energy per unit volume                          \n
        """

        # Set density, u, v and pressure
        self.rho = rho.copy()
        self.rhou = rho * u
        self.rhov = rho * v
        self.e = p / (self.g - 1) + 0.5 * (u ** 2 + v ** 2) / rho

        # Set state vector from state variables
        self.set_state_from_vars()

    def to_primitive_state(self) -> PrimitiveState:
        """
        Creates a `PrimitiveState` object from itself
        """
        return PrimitiveState(self.inputs, U_vector=self.U)

    def H(self):
        return self.g / (self.g - 1) * self.p() / self.rho + 0.5 * (self.rhou**2 + self.rhov**2) / self.rho

    def a(self):
        return np.sqrt(self.g * self.p() / self.rho)

    def u(self):
        return self.rhou / self.rho

    def v(self):
        return self.rhov / self.rho

    def p(self):
        return (self.g - 1) * (self.e - 0.5 * (self.rhou ** 2 + self.rhov ** 2) / self.rho)

    def non_dim(self):
        """
        Non-dimentionalizes the conservative state vector W. Let the non-dimentionalized conservative state vector be
        $U^* = \\begin{bmatrix} \\rho^* \\ \\rho u^* \\ \\rho v^* \\ e^* \\end{bmatrix}^T$, where $^*$ indicates a
        non-dimentionalized quantity. The non-dimentionalized conservative variables are:   \n
        $\\rho^* = \\rho/\\rho_\\infty$                                                     \n
        $\\rho u^* = \\rho u/a_\\infty \\rho_\\infty$                                       \n
        $\\rho u^* = \\rho v/a_\\infty \\rho_\\infty$                                       \n
        $e^* = e/\\rho_\\infty a_\\infty^2$                                                 \n
        If `ConservativeState` is created from a non-dimentionalized `PrimitiveState`, it will be non-dimentional.
        """

        self.U[:, :, 0] /= self.inputs.rho_inf
        self.U[:, :, 1] /= self.inputs.rho_inf * self.inputs.a_inf
        self.U[:, :, 2] /= self.inputs.rho_inf * self.inputs.a_inf
        self.U[:, :, 3] /= self.inputs.rho_inf * self.inputs.a_inf ** 2

        self.set_vars_from_state()


class RoePrimitiveState(PrimitiveState):
    """
    #*$Roe$* Primitive State
    A class the represents the *Roe* average state in primitive form. This class extends `PrimitiveState`, and is used to
    compute the *Roe* average of two primitive states, usually the left and right solution vectors $(W^R, W^L)$ at the
    left and right cell boundaries in a finite volume method. The primitive *Roe* average state vector is
    $W^* = \\begin{bmatrix} \\rho^* \\ u^* \\ v^* \\ p^* \\end{bmatrix}^T$, where $\\rho^\\*$ is the *Roe* density,
    $u^\\*$ is the $x$-direction *Roe* momentum per unit volume, $v^\\*$ is the $y$-direction *Roe* momentum per unit volume,
    and $p^\\*$ is the *Roe* energy per unit volume. This directly represents the *Roe* average quantities that can be
    used in various approximate Riemann solvers such as *Roe, HLLE, HLLL,* etc. The *Roe* average is computed as follows: \n

    Given the left and right primitive solution vectors $(W^R, W^L)$, where
    $W^R = \\begin{bmatrix} \\rho^R \\ u^R \\ v^R \\ p^R \\end{bmatrix}^T$ and
    $W^L = \\begin{bmatrix} \\rho^L \\ u^L \\ v^L \\ p^L \\end{bmatrix}^T$, the components of $W^\\*$ are:  \n
    $\\rho^\\* = \\sqrt{\\rho^R \\rho^L}$,                                                                  \n
    $u^\\* = \\frac{u^R \\sqrt{\\rho^R} + u^L \\sqrt{\\rho^L}}{\\sqrt{\\rho^R} + \\sqrt{\\rho^L}}$,         \n
    $v^\\* = \\frac{v^R \\sqrt{\\rho^R} + v^L \\sqrt{\\rho^L}}{\\sqrt{\\rho^R} + \\sqrt{\\rho^L}}$,         \n
    $p^\\* = \\frac{\\gamma - 1}{\\gamma}\\rho^\\*[H^\\* - \\frac{1}{2}({u^\\*}^2 + {v^\\*}^2)]$,           \n
    where $H^\\*$ is the Roe specific enthalpy, evaluated as:                                               \n
    $H^\\* = \\frac{H^R \\sqrt{\\rho^R} + H^L \\sqrt{\\rho^L}}{\\sqrt{\\rho^R} + \\sqrt{\\rho^L}}$.
    """

    def __init__(self, inputs, WL, WR):
        """
        ## Attributes

        **Public**                                  \n
            W       Primitive state vector          \n
            rho     density                         \n
            u       x-direction velocity            \n
            v       y-direction velocity            \n
            p       pressure                        \n
        """

        # Call superclass constructor
        super().__init__(inputs, nx=WL.W.shape[1], ny=WL.W.shape[0])
        self.roe_state_from_primitive_states(WL, WR)


    def roe_state_from_primitive_states(self, WL: PrimitiveState, WR: PrimitiveState) -> None:
        """
        Compute *Roe* average quantities of left and right primtive states

        **Parameters**                                              \n
            WL      Left primitive state                            \n
            WR      Right primitive state                           \n
        """

        # Get average *Roe* quantities
        rho, u, v, p = self.get_roe_state(WL, WR)

        # Set state vector and variables from *Roe* averages
        self.from_primitive_state_vars(rho, u, v, p)

    def roe_state_from_conservative_states(self, UL: ConservativeState, UR: ConservativeState) -> None:
        """
         Compute *Roe* average quantities of left and right conservative states

         **Parameters**                                              \n
             UL      Left conservative state                         \n
             UR      Right conservative state                        \n
         """

        # Get average *Roe* quantities
        rho, u, v, p = self.get_roe_state(UL.to_primitive_state(), UR.to_primitive_state())

        # Set state vector and variables from *Roe* averages
        self.from_primitive_state_vars(rho, u, v, p)

    def get_roe_state(self, WL: PrimitiveState, WR: PrimitiveState) -> [np.ndarray]:
        """
        Compute *Roe* average quantities as described earlier

        **Parameters**                                              \n
            WL      Left primitive state                            \n
            WR      Right primitive state                           \n
        """

        # Compute common quantities
        sqRhoL  = np.sqrt(WL.rho)
        sqRhoR  = np.sqrt(WR.rho)
        sqRhoRL = sqRhoL + sqRhoR

        # Compute *Roe* average quantities
        rho     = np.sqrt(WL.rho * WR.rho)
        u       = (WL.u * sqRhoL + WR.u * sqRhoR) / sqRhoRL
        v       = (WL.v * sqRhoL + WR.v * sqRhoR) / sqRhoRL
        H       = (WL.H() * sqRhoL + WR.H() * sqRhoR) / sqRhoRL
        p       = (self.g - 1) / self.g * rho * (H - 0.5 * (u**2 + v**2))

        return rho, u, v, p
