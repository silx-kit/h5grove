from typing import TypedDict, Union
import h5py

H5pyEntity = Union[
    h5py.Dataset, h5py.Datatype, h5py.ExternalLink, h5py.Group, h5py.SoftLink
]


class EntityMetadata(TypedDict):
    name: str
    type: str
