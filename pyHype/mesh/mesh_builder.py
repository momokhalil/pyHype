from pyHype.mesh import meshs as meshs
from numba.typed import Dict as nDict
from numba.typed import List as nList
from numba.experimental import jitclass
from numba.core.types import ListType
from numba.core.types import string as nstr, int32

_MESH_INPUT_SPEC = [('nBLK', int32),
                    ('NE', ListType(int32)),
                    ('NW', ListType(int32)),
                    ('SE', ListType(int32)),
                    ('SW', ListType(int32)),
                    ('n', int32),
                    ('nx', int32),
                    ('ny', int32),
                    ('NeighborE', int32),
                    ('NeighborW', int32),
                    ('NeighborN', int32),
                    ('NeighborS', int32),
                    ('BCTypeE', nstr),
                    ('BCTypeW', nstr),
                    ('BCTypeN', nstr),
                    ('BCTypeS', nstr)]


@jitclass(_MESH_INPUT_SPEC)
class BlockDescription:
    def __init__(self, list_, int_, str_):

        # Set parameter attributes from input dict
        self.nBLK = int_['nBLK']
        self.n = int_['n']
        self.nx = int_['nx']
        self.ny = int_['ny']
        self.NeighborE = int_['NeighborE']
        self.NeighborW = int_['NeighborW']
        self.NeighborN = int_['NeighborN']
        self.NeighborS = int_['NeighborS']
        self.NE = list_['NE']
        self.NW = list_['NW']
        self.SE = list_['SE']
        self.SW = list_['SW']
        self.BCTypeE = str_['BCTypeE']
        self.BCTypeW = str_['BCTypeW']
        self.BCTypeN = str_['BCTypeN']
        self.BCTypeS = str_['BCTypeS']


_block_description_type = BlockDescription.class_type.instance_type


def _to_nDict(dict_: dict):

    _lst = ListType(int32)
    _list_items = nDict.empty(key_type=nstr, value_type=_lst)
    _int_items = nDict.empty(key_type=nstr, value_type=int32)
    _str_items = nDict.empty(key_type=nstr, value_type=nstr)

    for key, value in dict_.items():
        if isinstance(value, list):
            _list_items[key] = nList(value)
        elif isinstance(value, int):
            _int_items[key] = value
        elif isinstance(value, str):
            _str_items[key] = value
        else:
            print('BRUH')
    return _list_items, _int_items, _str_items


def make_mesh_inputs(*args):
    _mesh = {}
    for arg in args:
        _mesh[arg['nBLK']] = arg
    return _mesh


# MESH INPUTS BUILDER
def build(mesh_name: str, nx: int, ny: int):

    _mesh = {}
    n = nx * ny

    if mesh_name == 'one_mesh':
        _mesh = meshs.one_mesh(n, nx, ny)
    elif mesh_name == 'simple_mesh':
        _mesh = meshs.simple_mesh(n, nx, ny)
    else:
        raise ValueError('Specified mesh name does not exist')

    mesh = nDict.empty(key_type=int32, value_type=_block_description_type)

    for nBLK, block in _mesh.items():
        list_, int_, str_ = _to_nDict(block)
        mesh[nBLK] = BlockDescription(list_, int_, str_)

    return mesh
