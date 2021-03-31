from pyHype.mesh import mesh_builder

# Meshes
def simple_mesh(nx, ny, n):
    block1 = {'nBLK': 1,
              'NE': [5, 5],
              'NW': [0, 5],
              'SE': [5, 0],
              'SW': [0, 0],
              'nx': nx,
              'ny': ny,
              'n': n,
              'NeighborE': 4,
              'NeighborW': 0,
              'NeighborN': 2,
              'NeighborS': 0,
              'BCTypeE': 'None',
              'BCTypeW': 'Reflection',
              'BCTypeN': 'None',
              'BCTypeS': 'Reflection'}

    block2 = {'nBLK': 2,
              'NE': [5, 10],
              'NW': [0, 10],
              'SE': [5, 5],
              'SW': [0, 5],
              'nx': nx,
              'ny': ny,
              'n': n,
              'NeighborE': 3,
              'NeighborW': 0,
              'NeighborN': 0,
              'NeighborS': 1,
              'BCTypeE': 'None',
              'BCTypeW': 'Reflection',
              'BCTypeN': 'Reflection',
              'BCTypeS': 'None'}

    block3 = {'nBLK': 3,
              'NE': [10, 10],
              'NW': [5, 10],
              'SE': [10, 5],
              'SW': [5, 5],
              'nx': nx,
              'ny': ny,
              'n': n,
              'NeighborE': 0,
              'NeighborW': 2,
              'NeighborN': 0,
              'NeighborS': 4,
              'BCTypeE': 'Reflection',
              'BCTypeW': 'None',
              'BCTypeN': 'Reflection',
              'BCTypeS': 'None'}

    block4 = {'nBLK': 4,
              'NE': [10, 5],
              'NW': [5, 5],
              'SE': [10, 0],
              'SW': [5, 0],
              'nx': nx,
              'ny': ny,
              'n': n,
              'NeighborE': 0,
              'NeighborW': 1,
              'NeighborN': 3,
              'NeighborS': 0,
              'BCTypeE': 'Reflection',
              'BCTypeW': 'None',
              'BCTypeN': 'None',
              'BCTypeS': 'Reflection'}

    return mesh_builder.make_mesh_inputs(block1, block2, block3, block4)

def one_mesh(n, nx, ny):
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

    return mesh_builder.make_mesh_inputs(block1)