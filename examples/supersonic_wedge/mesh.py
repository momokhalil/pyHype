import numpy as np
from pyhype.states import PrimitiveState
from pyhype.boundary_conditions.base import PrimitiveDirichletBC


def make_supersonic_wedge_mesh(fluid):
    # Define inlet BC
    inlet_rho = 1.0
    inlet_u = 2.0
    inlet_v = 0.0
    inlet_p = 1 / fluid.gamma()

    inlet_state = PrimitiveState(
        fluid=fluid,
        array=np.array(
            [
                inlet_rho,
                inlet_u,
                inlet_v,
                inlet_p,
            ]
        ).reshape((1, 1, 4)),
    )

    mach_2_wedge_inlet_bc = PrimitiveDirichletBC(primitive_state=inlet_state)

    # Define mesh
    block1 = {
        "nBLK": 1,
        "NW": [0, 2],
        "NE": [2, 2],
        "SW": [0, 0],
        "SE": [2, 0],
        "NeighborE": 2,
        "NeighborW": None,
        "NeighborN": None,
        "NeighborS": None,
        "NeighborNE": None,
        "NeighborNW": None,
        "NeighborSE": None,
        "NeighborSW": None,
        "BCTypeE": None,
        "BCTypeW": mach_2_wedge_inlet_bc,
        "BCTypeN": "OutletDirichlet",
        "BCTypeS": "Reflection",
        "BCTypeNE": None,
        "BCTypeNW": None,
        "BCTypeSE": None,
        "BCTypeSW": None,
    }
    block2 = {
        "nBLK": 2,
        "NW": [2, 2],
        "NE": [4, 2 + 2 * np.tan(15 * np.pi / 180)],
        "SW": [2, 0],
        "SE": [4, 2 * np.tan(15 * np.pi / 180)],
        "NeighborE": None,
        "NeighborW": 1,
        "NeighborN": None,
        "NeighborS": None,
        "NeighborNE": None,
        "NeighborNW": None,
        "NeighborSE": None,
        "NeighborSW": None,
        "BCTypeE": "OutletDirichlet",
        "BCTypeW": None,
        "BCTypeN": "OutletDirichlet",
        "BCTypeS": "Reflection",
        "BCTypeNE": None,
        "BCTypeNW": None,
        "BCTypeSE": None,
        "BCTypeSW": None,
    }
    return {
        1: block1,
        2: block2,
    }
