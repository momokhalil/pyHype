from numba.core.types import int32, float64
from pyHype.input.input_file_builder import propblem_input_type


STATE_SPEC = [('inputs', propblem_input_type),
              ('_size', int32),
              ('g', float64)]

PRIMITIVESTATE_SPEC = [('inputs', propblem_input_type),
                       ('_size', int32),
                       ('g', float64),
                       ('W', float64[:, :]),
                       ('rho', float64[:, :]),
                       ('u', float64[:, :]),
                       ('v', float64[:, :]),
                       ('p', float64[:, :])]

CONSERVATIVESTATE_SPEC = [('inputs', propblem_input_type),
                          ('_size', int32),
                          ('g', float64),
                          ('U', float64[:, :]),
                          ('rho', float64[:, :]),
                          ('rhou', float64[:, :]),
                          ('rhov', float64[:, :]),
                          ('e', float64[:, :])]

ROEPRIMITIVESTATE_SPEC = [('inputs', propblem_input_type),
                          ('_size', int32),
                          ('g', float64),
                          ('W', float64[:, :]),
                          ('rho', float64[:, :]),
                          ('u', float64[:, :]),
                          ('v', float64[:, :]),
                          ('p', float64[:, :])]
