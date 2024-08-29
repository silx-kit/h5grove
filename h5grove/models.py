from __future__ import annotations
from enum import Enum
from typing import Union, Tuple, Dict, List
from typing_extensions import TypedDict, NotRequired
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
# Must use functional `TypedDict` syntax because of `class` key
TypeMetadata = TypedDict(
    "TypeMetadata",
    {
        "class": int,  # HDF5 class code
        "dtype": StrDtype,  # Numpy-style dtype
        "size": int,  # all (but most relevant for int, float, string)
        "order": int,  # int, float, bitfield
        "sign": int,  # int
        "cset": int,  # string
        "strpad": int,  # string
        "vlen": bool,  # string
        "tag": str,  # opaque
        "dims": Tuple[int, ...],  # array
        "members": Union[Dict[str, "TypeMetadata"], Dict[str, int]],  # compound, enum
        "base": "TypeMetadata",  # array, enum, vlen
    },
    total=False,
)


class EntityMetadata(TypedDict):
    name: str
    kind: str


class ExternalLinkMetadata(EntityMetadata):
    target_file: str
    target_path: str


class SoftLinkMetadata(EntityMetadata):
    target_path: str


class AttributeMetadata(TypedDict):
    name: str
    shape: tuple
    type: TypeMetadata


class ResolvedEntityMetadata(EntityMetadata):
    attributes: List[AttributeMetadata]


class GroupMetadata(ResolvedEntityMetadata):
    children: NotRequired[List[EntityMetadata]]


class DatasetMetadata(ResolvedEntityMetadata):
    chunks: tuple
    filters: tuple
    shape: tuple
    type: TypeMetadata


class DatatypeMetadata(ResolvedEntityMetadata):
    type: TypeMetadata


class Stats(TypedDict):
    strict_positive_min: Union[int, float, None]
    positive_min: Union[int, float, None]
    min: Union[int, float, None]
    max: Union[int, float, None]
    mean: Union[int, float, None]
    std: Union[int, float, None]
