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
              'NeighborW': None,
              'NeighborN': 2,
              'NeighborS': None,
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
              'NeighborW': None,
              'NeighborN': None,
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
              'NeighborE': None,
              'NeighborW': 2,
              'NeighborN': None,
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
              'NeighborE': None,
              'NeighborW': 1,
              'NeighborN': 3,
              'NeighborS': None,
              'BCTypeEast': 'Reflection',
              'BCTypeWest': 'None',
              'BCTypeNorth': 'None',
              'BCTypeSouth': 'Reflection'}

    mesh = {1: block1,
            2: block2,
            3: block3,
            4: block4}

    return mesh

def one_mesh(nx, ny, n):
    block1 = {'nBLK': 1,
              'NE': (10, 10),
              'NW': (0, 10),
              'SE': (10, 0),
              'SW': (0, 0),
              'nx': nx,
              'ny': ny,
              'n': n,
              'NeighborE': None,
              'NeighborW': None,
              'NeighborN': None,
              'NeighborS': None,
              'BCTypeEast': 'Reflection',
              'BCTypeWest': 'Reflection',
              'BCTypeNorth': 'Reflection',
              'BCTypeSouth': 'Reflection'}

    mesh = {1: block1}

    return mesh
