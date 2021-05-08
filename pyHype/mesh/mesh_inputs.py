import pyHype.mesh.meshes as meshes

class BlockDescription:
    def __init__(self, blk_input):

        # Set parameter attributes from input dict
        self.nBLK = blk_input['nBLK']
        self.n = blk_input['n']
        self.nx = blk_input['nx']
        self.ny = blk_input['ny']
        self.NeighborE = blk_input['NeighborE']
        self.NeighborW = blk_input['NeighborW']
        self.NeighborN = blk_input['NeighborN']
        self.NeighborS = blk_input['NeighborS']
        self.NE = blk_input['NE']
        self.NW = blk_input['NW']
        self.SE = blk_input['SE']
        self.SW = blk_input['SW']
        self.BCTypeE = blk_input['BCTypeE']
        self.BCTypeW = blk_input['BCTypeW']
        self.BCTypeN = blk_input['BCTypeN']
        self.BCTypeS = blk_input['BCTypeS']


def build(mesh_name, nx, ny):

    mesh_func = meshes.DEFINED_MESHES[mesh_name]
    mesh_dict = mesh_func(nx=nx, ny=ny)
    mesh = {}

    for blk, blkData in mesh_dict.items():
        mesh[blk] = BlockDescription(blkData)

    return mesh
