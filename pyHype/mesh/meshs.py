from pyHype.mesh import mesh_builder

# Meshes
def simple_mesh(nx, ny, n):
    block1 = {'nBLK': 1,
              'NE': (5, 5),
              'NW': (0, 5),
              'SE': (5, 0),
              'SW': (0, 0),
              'nx': nx,
              'ny': ny,
              'n': n,
              'NeighborE': 4,
              'NeighborW': 0,
              'NeighborN': 2,
              'NeighborS': 0,
              'BCTypeEast': 'None',
              'BCTypeWest': 'Reflection',
              'BCTypeNorth': 'None',
              'BCTypeSouth': 'Reflection'}

    block2 = {'nBLK': 2,
              'NE': (5, 10),
              'NW': (0, 10),
              'SE': (5, 5),
              'SW': (0, 5),
              'nx': nx,
              'ny': ny,
              'n': n,
              'NeighborE': 3,
              'NeighborW': 0,
              'NeighborN': 0,
              'NeighborS': 1,
              'BCTypeEast': 'None',
              'BCTypeWest': 'Reflection',
              'BCTypeNorth': 'Reflection',
              'BCTypeSouth': 'None'}

    block3 = {'nBLK': 3,
              'NE': (10, 10),
              'NW': (5, 10),
              'SE': (10, 5),
              'SW': (5, 5),
              'nx': nx,
              'ny': ny,
              'n': n,
              'NeighborE': 0,
              'NeighborW': 2,
              'NeighborN': 0,
              'NeighborS': 4,
              'BCTypeEast': 'Reflection',
              'BCTypeWest': 'None',
              'BCTypeNorth': 'Reflection',
              'BCTypeSouth': 'None'}

    block4 = {'nBLK': 4,
              'NE': (10, 5),
              'NW': (5, 5),
              'SE': (10, 0),
              'SW': (5, 0),
              'nx': nx,
              'ny': ny,
              'n': n,
              'NeighborE': 0,
              'NeighborW': 1,
              'NeighborN': 3,
              'NeighborS': 0,
              'BCTypeEast': 'Reflection',
              'BCTypeWest': 'None',
              'BCTypeNorth': 'None',
              'BCTypeSouth': 'Reflection'}

    mesh = mesh_builder.build_numba_dict_for_mesh(block1, block2, block3, block4)

    return mesh

def one_mesh(nx, ny, n):
    block1 = {'nBLK': 1,
              'NE': [10, 10],
              'NW': [0, 10],
              'SE': [10, 0],
              'SW': [0, 0],
              'nx': nx,
              'ny': ny,
              'n': n,
              'NeighborE': 0,
              'NeighborW': 0,
              'NeighborN': 0,
              'NeighborS': 0,
              'BCTypeE': 'Reflection',
              'BCTypeW': 'Reflection',
              'BCTypeN': 'Reflection',
              'BCTypeS': 'Reflection'}

    mesh = mesh_builder.build_numba_dict_for_mesh(block1)

    return mesh
