import numpy as np
from pyhype.fluids import Air
from pyhype.states import PrimitiveState
from pyhype.mesh.base import QuadMeshGenerator
from pyhype.boundary_conditions.base import PrimitiveDirichletBC

# Define fluid
air = Air(a_inf=343.0, rho_inf=1.0)

inlet_rho = 1.0
inlet_u = 0.1
inlet_v = 0.0
inlet_p = 2.0 / air.gamma()

inlet_state = PrimitiveState(
    fluid=air,
    array=np.array(
        [
            inlet_rho,
            inlet_u,
            inlet_v,
            inlet_p,
        ]
    ).reshape((1, 1, 4)),
)
subsonic_inlet_bc = PrimitiveDirichletBC(primitive_state=inlet_state)

BCE = ["OutletDirichlet" for _ in range(9)]
BCW = [
    "Slipwall",
    "Slipwall",
    "Slipwall",
    "Slipwall",
    subsonic_inlet_bc,
    "Slipwall",
    "Slipwall",
    "Slipwall",
    "Slipwall",
]
BCN = ["OutletDirichlet"]
BCS = ["OutletDirichlet"]

mesh = QuadMeshGenerator(
    nx_blk=1,
    ny_blk=9,
    BCE=BCE,
    BCW=BCW,
    BCN=BCN,
    BCS=BCS,
    NE=(1, 0.5),
    SW=(0, 0),
    NW=(0, 0.5),
    SE=(1, 0),
)
