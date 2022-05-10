import io
from typing import Any, Callable, Dict, Optional, Union
import numpy as np
import orjson
import h5py
import tifffile

from .utils import is_numeric_data


def bin_encode(array: np.ndarray) -> bytes:
    """Convert array to bytes.

    :param array: Data to convert
    """
    return array.tobytes()


def orjson_default(o: Any) -> Union[list, float, str, None]:
    """Converts Python objects to JSON-serializable objects.

    :raises TypeError: if the object is not supported."""
    if isinstance(o, np.number) and o.dtype.kind == "f" and o.itemsize > 8:
        # Force conversion of float >64bits to native float even if it means losing precision
        return float(o)
    if isinstance(o, (np.generic, np.ndarray)):
        return o.tolist()
    if isinstance(o, complex):
        return [o.real, o.imag]
    if isinstance(o, h5py.Empty):
        return None
    if isinstance(o, bytes):
        return o.decode()
    if isinstance(o, (h5py.RegionReference, h5py.Reference)):
        return str(o)
    raise TypeError


def orjson_encode(content: Any, default: Optional[Callable] = None) -> bytes:
    """Encode in JSON using orjson.

    :param: content: Content to encode
    :param default: orjson default (https://github.com/ijl/orjson#default)
    """
    if default is None:
        default = orjson_default

    return orjson.dumps(content, default=default, option=orjson.OPT_SERIALIZE_NUMPY)


def csv_encode(data: np.ndarray) -> bytes:
    """Encodes a NumPy array in CSV.

    :param: data: NumPy array to encode
    """
    with io.BytesIO() as buffer:
        np.savetxt(buffer, data, delimiter=",")
        return buffer.getvalue()


def npy_encode(data: np.ndarray) -> bytes:
    """Encodes a NumPy array in NPY.

    :param: data: NumPy array to encode
    """
    with io.BytesIO() as buffer:
        np.save(buffer, data)
        return buffer.getvalue()


def tiff_encode(data: np.ndarray) -> bytes:
    """Encodes a NumPy array in TIFF. The data should be 2D.

    :param: data: NumPy array to encode
    """
    with io.BytesIO() as buffer:
        tifffile.imwrite(buffer, data, photometric="minisblack")
        return buffer.getvalue()


class Response:
    content: bytes
    """ Encoded `content` as bytes """
    headers: Dict[str, str]
    """ Associated headers """

    def __init__(self, content: bytes, headers: Dict[str, str]):
        self.content = content
        self.headers = {**headers, "Content-Length": str(len(content))}


def encode(content: Any, encoding: Optional[str] = "json") -> Response:
    """Encode content in given encoding.

    Warning: Not all encodings supports all types of content.

    :param content: Content to encode
    :param encoding:
        - `json` (default)
        - `bin`: nD array/scalars in bytes
        - `csv`: nD arrays in downloadable csv files
        - `npy`: nD arrays in downloadable npy files
        - `tiff`: 2D arrays in downloadable TIFF files
    :returns: A Response object containing content and headers
    :raises ValueError: If encoding is not among the ones above.
    """
    if encoding in ("json", None):
        return Response(
            orjson_encode(content),
            headers={"Content-Type": "application/json"},
        )

    content_array = np.array(content, copy=False)
    if not is_numeric_data(content_array):
        raise ValueError(f"Unsupported encoding {encoding} for non-numeric content")

    if encoding == "bin":
        return Response(
            bin_encode(content_array),
            headers={
                "Content-Type": "application/octet-stream",
            },
        )

    if encoding == "csv":
        return Response(
            csv_encode(content_array),
            headers={
                "Content-Type": "text/csv",
                "Content-Disposition": 'attachment; filename="data.csv"',
            },
        )

    if encoding == "npy":
        return Response(
            npy_encode(content_array),
            headers={
                "Content-Type": "application/octet-stream",
                "Content-Disposition": 'attachment; filename="data.npy"',
            },
        )

    if encoding == "tiff":
        return Response(
            tiff_encode(content_array),
            headers={
                "Content-Type": "image/tiff",
                "Content-Disposition": 'attachment; filename="data.tiff"',
            },
        )

    raise ValueError(f"Unsupported encoding {encoding}")
