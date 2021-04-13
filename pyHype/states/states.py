import numpy as np
from abc import abstractmethod
import pyHype.states.numba_spec as ns
from numba.experimental import jitclass


class State:
    """
    # State
    Defines an abstract class for implementing primitive and conservative state classes. The core components of a state
    are the state vector and the state variables. The state vector is composed of the state variables in a specific
    order. For example, for a state X with state variables $x_1, x_2, ..., x_n$ and state vector $X$, the state vector
    is represented as:
    $X = \\begin{bmatrix} x_1 \\ x_2 \\ \\dots \\ x_n \\end{bmatrix}^T$. The state vector represents the solution at
    each physical discretization point.
    """
    def __init__(self, inputs, size_):
        """
        ## Attributes

        **Private**                                 \n
            input       input dictionary            \n
            size        size of grid in block       \n

        **Public**                                  \n
            g           (gamma) specific heat ratio \n
        """

        # Private
        self.inputs = inputs
        self._size = size_

        # Public
        self.g = inputs.gamma

    @abstractmethod
    def set_state_from_vars(self, **kwargs):
        """
        Sets the state vector from present state variable attributes
        """
        pass

    @abstractmethod
    def set_vars_from_state(self, **kwargs):
        """
        Sets the state variable attributes from present state vector
        """
        pass

    @abstractmethod
    def from_vars(self, **kwargs):
        """
        Sets the state variable attributes and state vector from input state variables (passed as kwargs)
        """
        pass

    @abstractmethod
    def non_dim(self):
        """
        Makes state vector and state variables non-dimensional
        """
        pass

    @abstractmethod
    def a(self):
        """
        Returns speed of sound
        """
        pass

    @abstractmethod
    def H(self):
        """
        Returns total entalpy
        """
        pass


@jitclass(ns.PRIMITIVESTATE_SPEC)
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
    __super__ = State.__init__

    def __init__(self, inputs, size_: int):
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
        # super().__init__(inputs, size_)

        # Has to be done this way because numba doesnt understand super() yet
        self.__super__(inputs, size_)

        # Public
        self.W = np.zeros((4 * size_, 1))       # conservative state vector
        self.rho = np.zeros((size_, 1))         # density
        self.u = np.zeros((size_, 1))           # x-direction velocity
        self.v = np.zeros((size_, 1))           # y-direction velocity
        self.p = np.zeros((size_, 1))           # pressure

        self.set_vars_from_state()

    # PRIVATE METHODS --------------------------------------------------------------------------------------------------

    def set_vars_from_state(self):
        """
        Sets primitive variables from primitive state vector
        """
        self.rho    = self.W[0::4]
        self.u      = self.W[1::4]
        self.v      = self.W[2::4]
        self.p      = self.W[3::4]

    def set_state_from_vars(self):
        """
        Sets primitive variables from primitive state vector
        """
        self.W[0::4] = self.rho
        self.W[1::4] = self.u
        self.W[2::4] = self.v
        self.W[3::4] = self.p

    # PUBLIC METHODS ---------------------------------------------------------------------------------------------------

    def from_vars(self, rho: np.ndarray, u: np.ndarray, v: np.ndarray, p: np.ndarray) -> None:
        """
        Populates the primitive state vector and variables W with their appropriate components

        **Parameters**                              \n
            rho     density                         \n
            u       x-direction velocity            \n
            v       y-direction velocity            \n
            p       pressure                        \n
        """

        # Set density, u, v and pressure
        self.rho = rho
        self.u = u
        self.v = v
        self.p = p

        # Set W components appropriately
        self.set_state_from_vars()

    def from_state_vector(self, W: 'PrimitiveState') -> None:
        self.W = W
        self.set_vars_from_state()

    def from_U(self, U: 'ConservativeState') -> None:
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
        self.rho = U.rho
        self.u = U.rhou / U.rho
        self.v = U.rhov / U.rho
        self.p = (self.g - 1) * (U.e - 0.5 * (U.rhou**2 + U.rhov**2) / U.rho)

        self.set_state_from_vars()

    def to_U(self) -> 'ConservativeState':
        """
        Creates a `ConservativeState` object from itself
        """
        U = ConservativeState(self.inputs, self._size)
        U.from_W(self)
        return U

    def H(self) -> np.ndarray:
        return (self.g / (self.g - 1)) * (self.p / self.rho) + np.mean(self.u**2 + self.v**2)

    def a(self) -> np.ndarray:
        #print(self.p)
        #print(self.rho)
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
        self.W[0::4] /= self.inputs.rho_inf
        self.W[1::4] /= self.inputs.a_inf
        self.W[2::4] /= self.inputs.a_inf
        self.W[3::4] /= self.inputs.rho_inf * self.inputs.a_inf ** 2

        # Set variables from non-dimensionalized W
        self.set_vars_from_state()


@jitclass(ns.CONSERVATIVESTATE_SPEC)
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
    super_ = State.__init__

    def __init__(self, inputs, size_: int):
        """
        ## Attributes

        **Public**                                                  \n
            U       Conservative state vector                       \n
            rho     density                                         \n
            rhou    x-direction momentum per unit volume            \n
            rhov    y-direction momentum per unit volume            \n
            e       energy per unit volume                          \n
        """

        # Call superclass constructor
        # super().__init__(inputs, size_)

        # Has to be done this way because numba doesnt understand super() yet
        self.super_(inputs, size_)

        # Public
        self.U = np.zeros((4 * size_, 1))           # conservative state vector
        self.rho = np.zeros((size_, 1))             # density
        self.rhou = np.zeros((size_, 1))            # x-direction momentum per unit volume
        self.rhov = np.zeros((size_, 1))            # y-direction momentum per unit volume
        self.e = np.zeros((size_, 1))               # energy per unit volume

        self.set_vars_from_state()

    # PRIVATE METHODS --------------------------------------------------------------------------------------------------

    def set_vars_from_state(self):
        """
        Sets conservative variables from conservativestate vector
        """
        self.rho    = self.U[0::4]
        self.rhou   = self.U[1::4]
        self.rhov   = self.U[2::4]
        self.e      = self.U[3::4]

    def set_state_from_vars(self):
        """
        Sets conservative variables from conservative state vector
        """
        self.U[0::4] = self.rho
        self.U[1::4] = self.rhou
        self.U[2::4] = self.rhov
        self.U[3::4] = self.e

    # Compute pressure
    def _p(self):
        return (self.g - 1) * (self.e - 0.5 * (self.rhou**2 + self.rhou**2) / self.rho)

    # PUBLIC METHODS ---------------------------------------------------------------------------------------------------

    def from_vars(self, rho=None, rhou=None, rhov=None, e=None):
        """
        Populates the primitive state vector and variables W with their appropriate components

        **Parameters**                                              \n
            rho     density                                         \n
            rhou    x-direction momentum per unit volume            \n
            rhov    y-direction momentum per unit volume            \n
            e       energy per unit volume                          \n
        """

        # Set density, u, v and pressure
        self.rho = rho
        self.rhou = rhou
        self.rhov = rhov
        self.e = e

        # Set state vector from state variables
        self.set_state_from_vars()

    def from_state_vector(self, U: 'ConservativeState') -> None:
        self.U = U
        self.set_vars_from_state()

    def from_W(self, W: [PrimitiveState, 'RoePrimitiveState']):
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
        self.rho    = W.rho
        self.rhou   = W.rho * W.u
        self.rhov   = W.rho * W.v
        self.e      = W.p / (self.g - 1) + 0.5 * W.rho * (W.u**2 + W.v**2)

        self.set_state_from_vars()

    def to_W(self) -> PrimitiveState:
        """
        Creates a `PrimitiveState` object from itself
        """
        W = PrimitiveState(self.inputs, self._size)
        W.from_U(self)

        return W

    def H(self):
        return self.g / (self.g - 1) * self._p() / self.rho + 0.5 * (self.rhou**2 + self.rhov**2) / self.rho

    def a(self):
        return np.sqrt(self.g * self._p() / self.rho)

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

        self.U[0::4] /= self.inputs.rho_inf
        self.U[1::4] /= self.inputs.rho_inf * self.inputs.a_inf
        self.U[2::4] /= self.inputs.rho_inf * self.inputs.a_inf
        self.U[3::4] /= self.inputs.rho_inf * self.inputs.a_inf ** 2

        self.set_vars_from_state()

    def F(self):
        F = np.zeros((4 * self.inputs.nx + 4, 1))

        F[0::4] = self.rhou
        F[1::4] = self._p() + self.rhou ** 2 / self.rho
        F[2::4] = self.rhou * self.rhov / self.rho
        F[3::4] = (self.rhou / self.rho) * (self.e + self._p())

        return F

    def G(self):
        G = np.zeros((4 * self.inputs.ny + 4, 1))

        G[0::4] = self.rhov
        G[1::4] = self.rhou * self.rhov / self.rho
        G[2::4] = self._p() + self.rhov ** 2 / self.rho
        G[3::4] = (self.rhov / self.rho) * (self.e + self._p())

        return G


@jitclass(ns.ROEPRIMITIVESTATE_SPEC)
class RoePrimitiveState(State):
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
    super_ = State.__init__

    def __init__(self, inputs, WL, WR, size_: int = None):
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
        # super().__init__(inputs, size_)

        # Has to be done this way because numba doesnt understand super() yet
        self.super_(inputs, size_)

        # Public
        self.W = np.zeros((4 * size_, 1))           # primitive state vector
        self.rho = np.zeros((size_, 1))         # density
        self.u = np.zeros((size_, 1))           # x-direction velocity
        self.v = np.zeros((size_, 1))           # y-direction velocity
        self.p = np.zeros((size_, 1))           # pressure

        self.roe_state_from_primitive_states(WL, WR)

    # PRIVATE METHODS --------------------------------------------------------------------------------------------------

    def set_vars_from_state(self):
        """
        Sets primitive variables from primitive state vector
        """
        self.rho    = self.W[0::4]
        self.u      = self.W[1::4]
        self.v      = self.W[2::4]
        self.p      = self.W[3::4]

    def set_state_from_vars(self):
        """
        Sets primitive variables from primitive state vector
        """
        self.W[0::4] = self.rho
        self.W[1::4] = self.u
        self.W[2::4] = self.v
        self.W[3::4] = self.p

    # PUBLIC METHODS ---------------------------------------------------------------------------------------------------

    def from_vars(self, rho: np.ndarray, u: np.ndarray, v: np.ndarray, p: np.ndarray) -> None:
        """
        Populates the primitive state vector and variables W with their appropriate components

        **Parameters**                              \n
            rho     density                         \n
            u       x-direction velocity            \n
            v       y-direction velocity            \n
            p       pressure                        \n
        """

        # Set density, u, v and pressure
        self.rho = rho
        self.u = u
        self.v = v
        self.p = p

        # Set W components appropriately
        self.set_state_from_vars()

    def from_state_vector(self, W: 'PrimitiveState') -> None:
        self.W = W
        self.set_vars_from_state()

    def from_U(self, U: 'ConservativeState') -> None:
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
        self.rho = U.rho
        self.u = U.rhou / U.rho
        self.v = U.rhov / U.rho
        self.p = (self.g - 1) * (U.e - 0.5 * (U.rhou**2 + U.rhov**2) / U.rho)

        self.set_state_from_vars()

    def to_U(self) -> 'ConservativeState':
        """
        Creates a `ConservativeState` object from itself
        """
        U = ConservativeState(self.inputs, self._size)
        U.from_W(self)
        return U

    def H(self) -> np.ndarray:
        return (self.g / (self.g - 1)) * (self.p / self.rho) + np.mean(self.u**2 + self.v**2)

    def a(self) -> np.ndarray:
        #print(self.p)
        #print(self.rho)
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
        self.W[0::4] /= self.inputs.rho_inf
        self.W[1::4] /= self.inputs.a_inf
        self.W[2::4] /= self.inputs.a_inf
        self.W[3::4] /= self.inputs.rho_inf * self.inputs.a_inf ** 2

        # Set variables from non-dimensionalized W
        self.set_vars_from_state()

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
        self.from_vars(rho, u, v, p)

    def roe_state_from_conservative_states(self, UL: ConservativeState, UR: ConservativeState) -> None:
        """
         Compute *Roe* average quantities of left and right conservative states

         **Parameters**                                              \n
             UL      Left conservative state                         \n
             UR      Right conservative state                        \n
         """

        # Get average *Roe* quantities
        rho, u, v, p = self.get_roe_state(UL.to_W(), UR.to_W())

        # Set state vector and variables from *Roe* averages
        self.from_vars(rho, u, v, p)

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
