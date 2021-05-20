import h5py
from numbers import Number
import numpy as np
from typing import Any, Optional, Sequence, Tuple, Union
from .models import H5pyEntity


def attrMetaDict(attrId):
    return {"dtype": attrId.dtype.str, "name": attrId.name, "shape": attrId.shape}


def get_entity_from_file(h5file: h5py.File, path: Optional[str] = None) -> H5pyEntity:
    if path is None:
        path = "/"

    if path == "/":
        return h5file[path]

    link = h5file.get(path, getlink=True)
    if isinstance(link, h5py.ExternalLink) or isinstance(link, h5py.SoftLink):
        try:
            return h5file[path]
        except (OSError, KeyError):
            return link

    return h5file[path]


def parse_slice(dataset: h5py.Dataset, slice_str: str) -> Tuple[Union[slice, int], ...]:
    if "," not in slice_str:
        return (parse_slice_member(slice_str, dataset.shape[0]),)

    slice_members = slice_str.split(",")

    if len(slice_members) > dataset.ndim:
        raise TypeError(
            f"{slice_str} is a {len(slice_members)}d slice while the dataset is {dataset.ndim}d"
        )

    return tuple(
        parse_slice_member(s, dataset.shape[i]) for i, s in enumerate(slice_members)
    )


def parse_slice_member(slice_member: str, max_dim: int) -> Union[slice, int]:
    if ":" not in slice_member:
        return int(slice_member)

    slice_params = slice_member.split(":")
    if len(slice_params) == 2:
        start, stop = slice_params

        return slice(
            int(start) if start != "" else 0, int(stop) if stop != "" else max_dim
        )

    if len(slice_params) == 3:
        start, stop, step = slice_params

        return slice(
            int(start) if start != "" else 0,
            int(stop) if stop != "" else max_dim,
            int(step) if step != "" else 1,
        )

    raise TypeError(f"{slice_member} is not a valid slice")


def sorted_dict(*args: Tuple[str, Any]):
    return dict(sorted(args))


def sanitize_array(array: Sequence[Number], copy: bool = True) -> np.ndarray:
    """Ensure array save as .npy can be read back by js-numpy-parser.

    See https://github.com/ludwigschubert/js-numpy-parser

    :param array: Array to sanitize
    :param copy: Set to False to avoid copy if possible
    :raises ValueError: For unsupported array dtype
    """
    if not isinstance(array, np.ndarray):
        array = np.array(array)

    if array.dtype.kind not in ("f", "i", "u"):
        raise ValueError("Unsupported array type")

    # Convert to little endian
    dtype = array.dtype.newbyteorder("little")

    if dtype.kind in ("i", "u"):
        if dtype.itemsize > 4:  # (u)int64 -> (u)int32
            dtype = np.dtype(f"<{dtype.kind}4")

    if dtype.kind == "f":
        if dtype.itemsize < 4:  # float16 -> float32
            dtype = np.dtype("<f4")
        elif dtype.itemsize > 8:  # float128 -> float64
            dtype = np.dtype("<f8")

    return np.array(array, copy=copy, order="C", dtype=dtype)
