from typing import Tuple, Union
import h5py

H5pyEntity = Union[
    h5py.Dataset, h5py.Datatype, h5py.ExternalLink, h5py.Group, h5py.SoftLink
]

Selection = Union[str, int, slice, Tuple[Union[slice, int], ...], None]
