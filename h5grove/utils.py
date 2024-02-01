from pathlib import Path
import h5py
from h5py.version import version_tuple as h5py_version
from os.path import basename
import numpy as np
from typing import Any, Callable, Dict, List, Optional, Tuple, TypeVar, Union

from .models import H5pyEntity, LinkResolution, Selection, StrDtype, TypeMetadata


class NotFoundError(Exception):
    pass


class PathError(NotFoundError):
    pass


class LinkError(NotFoundError):
    pass


class QueryArgumentError(ValueError):
    pass


def _get_attr_id(entity_attrs: h5py.AttributeManager, attr_name: str):
    return entity_attrs.get_id(attr_name)


def _legacy_get_attr_id(entity_attrs: h5py.AttributeManager, attr_name: str):
    return h5py.h5a.open(entity_attrs._id, entity_attrs._e(attr_name))


get_attr_id = (
    _legacy_get_attr_id
    if h5py_version.major <= 2 and h5py_version.minor <= 9
    else _get_attr_id
)


def attr_metadata(entity_attrs: h5py.AttributeManager, attr_name: str) -> dict:
    attrId = get_attr_id(entity_attrs, attr_name)

    return {
        "name": attr_name,
        "shape": attrId.shape,
        "type": get_type_metadata(attrId.get_type()),
    }


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


def parse_slice(slice_str: str) -> Tuple[Union[slice, int], ...]:
    """
    Parses a string containing a slice under NumPy format.

    Examples:
        '5' => (5,)
        '1, 2:5' => (1, slice(2,5))
        '0:10:5, 2, 3:' => (slice(0, 10, 5), 2, slice(3, None, None))

    :param slice_str: String containing the slice
    """
    if "," not in slice_str:
        return (parse_slice_member(slice_str),)

    slice_members = slice_str.split(",")

    return tuple(parse_slice_member(s) for s in slice_members)


def parse_slice_member(slice_member: str) -> Union[slice, int]:
    if ":" not in slice_member:
        return int(slice_member)

    slice_params = slice_member.split(":")
    if len(slice_params) == 2:
        start, stop = slice_params

        return slice(
            int(start) if start != "" else 0, int(stop) if stop != "" else None
        )

    if len(slice_params) == 3:
        start, stop, step = slice_params

        return slice(
            int(start) if start != "" else None,
            int(stop) if stop != "" else None,
            int(step) if step != "" else None,
        )

    raise TypeError(f"{slice_member} is not a valid slice")


def sorted_dict(*args: Tuple[str, Any]):
    return dict(sorted(args, key=lambda entry: entry[0]))


def get_type_metadata(type_id: h5py.h5t.TypeID) -> TypeMetadata:
    base_metadata: TypeMetadata = {
        "class": type_id.get_class(),
        "dtype": stringify_dtype(type_id.dtype),
        "size": type_id.get_size(),
    }
    members = {}

    if isinstance(type_id, h5py.h5t.TypeIntegerID):
        return {
            **base_metadata,
            "order": type_id.get_order(),
            "sign": type_id.get_sign(),
        }

    if isinstance(type_id, h5py.h5t.TypeFloatID):
        return {
            **base_metadata,
            "order": type_id.get_order(),
        }

    if isinstance(type_id, h5py.h5t.TypeStringID):
        return {
            **base_metadata,
            "cset": type_id.get_cset(),
            "vlen": type_id.is_variable_str(),
        }

    if isinstance(type_id, h5py.h5t.TypeBitfieldID):
        return {**base_metadata, "order": type_id.get_order()}

    if isinstance(type_id, h5py.h5t.TypeOpaqueID):
        return {**base_metadata, "tag": type_id.get_tag()}

    if isinstance(type_id, h5py.h5t.TypeCompoundID):
        for i in range(0, type_id.get_nmembers()):
            members[type_id.get_member_name(i).decode("utf-8")] = get_type_metadata(
                type_id.get_member_type(i)
            )

        return {**base_metadata, "members": members}

    if isinstance(type_id, h5py.h5t.TypeEnumID):
        for i in range(0, type_id.get_nmembers()):
            members[type_id.get_member_name(i).decode("utf-8")] = (
                type_id.get_member_value(i)
            )

        return {
            **base_metadata,
            "members": members,
            "base": get_type_metadata(type_id.get_super()),
        }

    if isinstance(type_id, h5py.h5t.TypeVlenID):
        return {**base_metadata, "base": get_type_metadata(type_id.get_super())}

    if isinstance(type_id, h5py.h5t.TypeArrayID):
        return {
            **base_metadata,
            "dims": type_id.get_array_dims(),
            "base": get_type_metadata(type_id.get_super()),
        }

    return base_metadata


def _sanitize_dtype(dtype: np.dtype) -> np.dtype:
    """Sanitize numpy dtype to one with a matching typed array in modern JavaScript.

    :raises ValueError: If trying to sanitize a non-numeric numpy dtype
    """
    if dtype.kind not in ("f", "i", "u"):
        raise ValueError(f"Unsupported numpy dtype `{dtype}`. Expected numeric dtype.")

    # Convert to little endian
    result = dtype.newbyteorder("<")

    # Convert float16 to float32
    if result.kind == "f" and result.itemsize < 4:
        return np.dtype("<f4")

    # Convert float128 to float64 (unavoidable loss of precision)
    if result.kind == "f" and result.itemsize > 8:
        return np.dtype("<f8")

    return result


T = TypeVar("T", np.ndarray, np.number, np.bool_)


def convert(data: T, dtype: Optional[str] = "origin") -> T:
    """Convert array or numpy scalar to given dtype query param

    :param data: nD array or scalar to convert
    :param dtype: Data type conversion query parameter
        - `origin` (default): No conversion
        - `safe`: Convert to type with matching JS TypedArray (https://developer.mozilla.org/fr/docs/Web/JavaScript/Reference/Global_Objects/TypedArray)

    :raises QueryArgumentError: When using `dtype=safe` to convert non-numeric data
    :raises QueryArgumentError: For unsupported `dtype` argument
    """
    if dtype in ("origin", None):
        return data

    if dtype == "safe":
        if not is_numeric_data(data):
            raise QueryArgumentError(f"Unsupported dtype {dtype} for non-numeric data")
        return data.astype(_sanitize_dtype(data.dtype), order="C", copy=False)

    raise QueryArgumentError(f"Unsupported dtype {dtype}")


def is_numeric_data(data: Union[np.ndarray, np.number, np.bool_, bytes]) -> bool:
    if not isinstance(data, (np.ndarray, np.number, np.bool_)):
        return False

    return np.issubdtype(data.dtype, np.number) or np.issubdtype(data.dtype, np.bool_)


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
        "strict_positive_min": (
            cast(np.min(strict_positive_data))
            if strict_positive_data.size != 0
            else None
        ),
        "positive_min": (
            cast(np.min(positive_data)) if positive_data.size != 0 else None
        ),
        "min": cast(np.min(data)),
        "max": cast(np.max(data)),
        "mean": cast(np.mean(data)),
        "std": cast(np.std(data)),
    }


def hdf_path_join(prefix: Union[str, None], suffix: str):
    if prefix is None or prefix == "/":
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

    raise QueryArgumentError(
        f"{raw_query_arg} is not a valid value for link resolution. Accepted values are: {LinkResolution.ALL}, {LinkResolution.NONE} or {LinkResolution.ONLY_VALID}"
    )


def get_dataset_slice(dataset: h5py.Dataset, selection: Selection):
    if selection is None:
        return dataset[()]

    if isinstance(selection, str):
        parsed_slice = parse_slice(selection)
        if len(parsed_slice) > dataset.ndim:
            raise ValueError(
                f"{selection} has too many members to slice a {dataset.ndim}D dataset"
            )
        return dataset[parsed_slice]

    return dataset[selection]


def get_filters(
    dataset: h5py.Dataset,
) -> Optional[List[Dict[str, Union[int, str]]]]:
    property_list = dataset.id.get_create_plist()

    n_filters = property_list.get_nfilters()
    if n_filters <= 0:
        return None

    return [get_filter_info(property_list.get_filter(i)) for i in range(n_filters)]


def get_filter_info(
    filter: Tuple[int, int, Tuple[int, ...], str]
) -> Dict[str, Union[int, str]]:
    # https://api.h5py.org/h5p.html#h5py.h5p.PropDCID.get_filter
    (filter_id, _, _, name) = filter

    return {"id": filter_id, "name": name}


def stringify_dtype(dtype: np.dtype) -> StrDtype:
    if dtype.fields is None:
        return dtype.str

    return {
        k: stringify_dtype(dtype_tuple[0]) for k, dtype_tuple in dtype.fields.items()
    }


def open_file_with_error_fallback(
    filepath: Union[str, Path],
    create_error: Callable[[int, str], Exception],
    h5py_options: Dict[str, Any] = {},
) -> h5py.File:
    try:
        f = h5py.File(filepath, "r", **h5py_options)
    except OSError as e:
        if isinstance(e, FileNotFoundError) or "No such file or directory" in str(e):
            raise create_error(404, "File not found!")
        if isinstance(e, PermissionError) or "Permission denied" in str(e):
            raise create_error(403, "Cannot read file: Permission denied!")
        raise e

    return f
