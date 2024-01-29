from enum import Enum
from typing import Dict, Tuple, Union
from typing_extensions import TypedDict
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


# Recursive types not supported by mypy: https://github.com/python/mypy/issues/731
StrDtype = Union[str, Dict[str, "StrDtype"]]  # type: ignore

# https://api.h5py.org/h5t.html
TypeMetadata = TypedDict(
    "TypeMetadata",
    {
        "class": int,  # HDF5 class code
        "dtype": StrDtype,  # Numpy-style dtype
        "size": int,  # all (but most relevant for int, float, string)
        "order": int,  # int, float, bitfield
        "sign": int,  # int
        "cset": int,  # string
        "vlen": bool,  # string
        "tag": str,  # opaque
        "dims": Tuple[int, ...],  # array
        "members": Union[Dict[str, "TypeMetadata"], Dict[str, int]],  # compound, enum
        "base": "TypeMetadata",  # array, enum, vlen
    },
    total=False,
)
