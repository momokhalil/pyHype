import numba
from pyHype.mesh import meshs as meshs
from numba.core.types import string as nstr
from numba.core.types import ListType
from numba.typed import Dict as nDict
from numba.typed import List as nList
from numba import int32, float32
from numba.experimental import jitclass

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


def build_numba_dict_for_mesh(*args):

    mesh = nDict.empty(key_type=int32, value_type=_block_description_type)

    for arg in args:
        list_, int_, str_ = dict_to_numbadict(arg)
        mesh[int_['nBLK']] = BlockDescription(list_, int_, str_)
    return mesh


def dict_to_numbadict(dict_: dict):

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
