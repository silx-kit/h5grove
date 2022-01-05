import h5py
from numbers import Number
from os.path import basename
import numpy as np
from typing import Any, Dict, Sequence, Tuple, Union

from .models import H5pyEntity, LinkResolution, Selection


class NotFoundError(Exception):
    pass


class PathError(NotFoundError):
    pass


class LinkError(NotFoundError):
    pass


def attr_metadata(attrId: h5py.h5a.AttrID) -> dict:
    return {"dtype": attrId.dtype.str, "name": attrId.name, "shape": attrId.shape}


def get_entity_from_file(
    h5file: h5py.File,
    path: str,
    resolve_links: LinkResolution = LinkResolution.ONLY_VALID,
) -> H5pyEntity:
    if path == "/":
        return h5file[path]

    link = h5file.get(path, getlink=True)

    if link is None:
        raise PathError(f"{path} is not a valid path in {basename(h5file.filename)}")

    if isinstance(link, h5py.ExternalLink) or isinstance(link, h5py.SoftLink):
        if resolve_links == LinkResolution.NONE:
            return link

        try:
            return h5file[path]
        except (OSError, KeyError):
            if resolve_links == LinkResolution.ONLY_VALID:
                return link

            raise LinkError(
                f"Cannot resolve {link} at {path} of {basename(h5file.filename)}"
            )

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


def _sanitize_dtype(dtype: np.dtype) -> np.dtype:
    """Convert dtype to a dtype supported by js-numpy-parser.

    See https://github.com/ludwigschubert/js-numpy-parser

    :raises ValueError: For unsupported array dtype
    """
    if dtype.kind not in ("f", "i", "u"):
        raise ValueError("Unsupported array type")

    # Convert to little endian
    result = dtype.newbyteorder("<")

    if result.kind == "i" and result.itemsize > 4:
        return np.dtype("<i4")  # int64 -> int32

    if result.kind == "u" and result.itemsize > 4:
        return np.dtype("<u4")  # uint64 -> uint32

    if result.kind == "f" and result.itemsize < 4:
        return np.dtype("<f4")

    if result.kind == "f" and result.itemsize > 8:
        return np.dtype("<f8")

    return result


def sanitize_array(array: Sequence[Number], copy: bool = True) -> np.ndarray:
    """Ensure array save as .npy can be read back by js-numpy-parser.

    See https://github.com/ludwigschubert/js-numpy-parser

    :param array: Array to sanitize
    :param copy: Set to False to avoid copy if possible
    :raises ValueError: For unsupported array dtype
    """
    ndarray = np.array(array, copy=False)
    return np.array(ndarray, copy=copy, order="C", dtype=_sanitize_dtype(ndarray.dtype))


def get_array_stats(data: np.ndarray) -> Dict[str, Union[float, int, None]]:
    if data.size == 0:
        return {
            "strict_positive_min": None,
            "positive_min": None,
            "min": None,
            "max": None,
            "mean": None,
            "std": None,
        }

    cast = float if np.issubdtype(data.dtype, np.floating) else int
    strict_positive_data = data[data > 0]
    positive_data = data[data >= 0]
    return {
        "strict_positive_min": cast(np.min(strict_positive_data))
        if strict_positive_data.size != 0
        else None,
        "positive_min": cast(np.min(positive_data))
        if positive_data.size != 0
        else None,
        "min": cast(np.min(data)),
        "max": cast(np.max(data)),
        "mean": cast(np.mean(data)),
        "std": cast(np.std(data)),
    }


def hdf_path_join(prefix, suffix):
    if prefix == "/":
        return f"/{suffix}"

    return f'{prefix.rstrip("/")}/{suffix}'


def parse_bool_arg(query_arg: Union[str, None], fallback: bool) -> bool:
    if query_arg is None:
        return fallback

    return query_arg.lower() != "false"


def parse_link_resolution_arg(
    raw_query_arg: Union[str, None], fallback: LinkResolution
) -> LinkResolution:
    if raw_query_arg is None:
        return fallback

    query_arg = raw_query_arg.lower()

    # Checking for "true"/"false" to keep the same behaviour as when the arg was a boolean
    if query_arg in ("true", LinkResolution.ALL):
        return LinkResolution.ALL

    if query_arg in ("false", LinkResolution.NONE):
        return LinkResolution.NONE

    if query_arg == LinkResolution.ONLY_VALID:
        return LinkResolution.ONLY_VALID

    raise ValueError(
        f"{raw_query_arg} is not a valid value for link resolution. Accepted values are: {LinkResolution.ALL}, f{LinkResolution.NONE} or {LinkResolution.ONLY_VALID}"
    )


def get_dataset_slice(dataset: h5py.Dataset, selection: Selection):
    if selection is None:
        return dataset[()]

    if isinstance(selection, str):
        return dataset[parse_slice(dataset, selection)]

    return dataset[selection]
