from pyhype.mesh.rectangular import RectagularMeshGenerator

mesh = RectagularMeshGenerator.generate(
    BCE=["Reflection"],
    BCW=["Reflection"],
    BCN=["Reflection"],
    BCS=["Reflection"],
    east=10.0,
    west=0.0,
    north=20.0,
    south=0.0,
    n_blocks_horizontal=2,
    n_blocks_vertical=4,
)
