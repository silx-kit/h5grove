from typing import Union

try:
    from typing import TypedDict
except ImportError:  # Python<3.8 support
    TypedDict = dict

import h5py

H5pyEntity = Union[
    h5py.Dataset, h5py.Datatype, h5py.ExternalLink, h5py.Group, h5py.SoftLink
]


class EntityMetadata(TypedDict):
    name: str
    type: str
