from pyHype.fvm.base import FiniteVolumeMethod
from pyHype.states import ConservativeState


class FirstOrderUnlimited(FiniteVolumeMethod):
    def __init__(self, input_, global_nBLK):
        super().__init__(input_, global_nBLK)

    def _get_slope(self, U): pass

    def _get_limiter(self, U): pass

    def _reconstruct_state_X(self, U):
        UL = ConservativeState(self._input, U=U[:-4])
        UR = ConservativeState(self._input, U=U[4:])

        self._flux_function_X.set_left_state(UL)
        self._flux_function_X.set_right_state(UR)

    def _reconstruct_state_Y(self, U):
        UL = ConservativeState(self._input, U=U[:-4])
        UR = ConservativeState(self._input, U=U[4:])

        self._flux_function_Y.set_left_state(UL)
        self._flux_function_Y.set_right_state(UR)


class SecondOrderLimited(FiniteVolumeMethod):
    def __init__(self, input_, global_nBLK):
        super().__init__(input_, global_nBLK)
        self._slope = None

    def _get_slope(self, U):
        slope = (U[8:] - U[4:-4]) / (U[4:-4] - U[:-8] + 1e-8)
        return slope * (slope > 0)

    def _get_limiter(self, U):
        slope = self._get_slope(U)
        return 0.5 * (self._flux_limiter(slope)) / (slope + 1)

    def _reconstruct_state_X(self, U):
        limited_state = self._get_limiter(U) * (U[8:] - U[:-8])
        left, right = U[:-4], U[4:]
        left[4:] += limited_state
        right[:-4] -= limited_state

        UL = ConservativeState(self._input, U=left)
        UR = ConservativeState(self._input, U=right)

        self._flux_function_X.set_left_state(UL=UL)
        self._flux_function_X.set_right_state(UR=UR)

    def _reconstruct_state_Y(self, U):
        limited_state = self._get_limiter(U) * (U[8:] - U[:-8])
        left, right = U[:-4], U[4:]
        left[4:] += limited_state
        right[:-4] -= limited_state

        UL = ConservativeState(self._input, U=left)
        UR = ConservativeState(self._input, U=right)

        self._flux_function_Y.set_left_state(UL=UL)
        self._flux_function_Y.set_right_state(UR=UR)
