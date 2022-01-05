from enum import Enum
from typing import Tuple, Union
import h5py

H5pyEntity = Union[
    h5py.Dataset, h5py.Datatype, h5py.ExternalLink, h5py.Group, h5py.SoftLink
]

Selection = Union[str, int, slice, Tuple[Union[slice, int], ...], None]


class LinkResolution(str, Enum):
    NONE = "none"  # Links are not resolved whatever their status
    ONLY_VALID = (
        "only_valid"  # Links are only resolved if valid and unresolved if broken
    )
    ALL = "all"  # Links are resolved no matter their status. The resolution of broken links raises LinkError
