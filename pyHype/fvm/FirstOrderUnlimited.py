from pyHype.fvm.base import FiniteVolumeMethod
from pyHype.states import ConservativeState

class FirstOrderUnlimited(FiniteVolumeMethod):
    def __init__(self, inputs, global_nBLK):
        super().__init__(inputs, global_nBLK)

    def _get_slope(self, U): pass

    def _get_limiter(self, U): pass

    def _reconstruct_state_X(self, U):
        UL = ConservativeState(self.inputs, U=U[:-4])
        UR = ConservativeState(self.inputs, U=U[4:])

        self._flux_function_X.set_left_state(UL)
        self._flux_function_X.set_right_state(UR)

    def _reconstruct_state_Y(self, U):
        UL = ConservativeState(self.inputs, U=U[:-4])
        UR = ConservativeState(self.inputs, U=U[4:])

        self._flux_function_Y.set_left_state(UL)
        self._flux_function_Y.set_right_state(UR)
