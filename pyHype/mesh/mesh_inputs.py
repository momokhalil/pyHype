import pyHype.mesh.meshes as meshes

class BlockDescription:
    def __init__(self, mesh_input):

        # Set parameter attributes from input dict
        self.nBLK = mesh_input['nBLK']
        self.n = mesh_input['n']
        self.nx = mesh_input['nx']
        self.ny = mesh_input['ny']
        self.NeighborE = mesh_input['NeighborE']
        self.NeighborW = mesh_input['NeighborW']
        self.NeighborN = mesh_input['NeighborN']
        self.NeighborS = mesh_input['NeighborS']
        self.NE = mesh_input['NE']
        self.NW = mesh_input['NW']
        self.SE = mesh_input['SE']
        self.SW = mesh_input['SW']
        self.BCTypeE = mesh_input['BCTypeE']
        self.BCTypeW = mesh_input['BCTypeW']
        self.BCTypeN = mesh_input['BCTypeN']
        self.BCTypeS = mesh_input['BCTypeS']

def build(mesh_name):
    mesh_dict = meshes.DEFINED_MESHES[mesh_name]
    return BlockDescription