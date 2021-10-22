import io
from numbers import Number
from typing import Any, Callable, Dict, Generator, NamedTuple, Optional, Sequence, Union
import numpy as np
import orjson
import h5py
from .utils import sanitize_array


def bin_encode(array: Sequence[Number]) -> bytes:
    """Sanitize an array and convert it to bytes.

    :param array: Data to convert
    """
    sanitized_array = sanitize_array(array)

    return sanitized_array.tobytes()


def orjson_default(o) -> Union[list, str, None]:
    """Converts Python objects to JSON-serializable objects.

    :raises TypeError: if the object is not supported."""
    if isinstance(o, np.generic) or isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, complex):
        return [o.real, o.imag]
    if isinstance(o, h5py.Empty):
        return None
    if isinstance(o, bytes):
        return o.decode()
    raise TypeError


def orjson_encode(content: Any, default: Optional[Callable] = None) -> bytes:
    """Encode in JSON using orjson.

    :param: content: Content to encode
    :param default: orjson default (https://github.com/ijl/orjson#default)
    """
    if default is None:
        default = orjson_default

    return orjson.dumps(content, default=default, option=orjson.OPT_SERIALIZE_NUMPY)


def npy_stream(array: Sequence[Number]) -> Generator[bytes, None, None]:
    """Generator to stream nD array as a .npy file.

    :param array: Data to stream
    """
    sanitized_array = sanitize_array(array)

    # Stream header
    with io.BytesIO() as buffer:
        np.lib.format.write_array_header_1_0(
            buffer, np.lib.format.header_data_from_array_1_0(sanitized_array)
        )
        header = buffer.getvalue()
    yield header

    # Taken from numpy.lib.format.write_array
    if sanitized_array.itemsize == 0:
        buffersize = 0
    else:
        # Set buffer size to 16 MiB to hide the Python loop overhead.
        buffersize = max(16 * 1024 ** 2 // sanitized_array.itemsize, 1)

    for chunk in np.nditer(
        sanitized_array,
        flags=["external_loop", "buffered", "zerosize_ok"],
        buffersize=buffersize,
        order="C",
    ):
        yield chunk.tobytes("C")


class Response(NamedTuple):
    content: Union[Generator[bytes, None, None], bytes]
    """ Encoded `content` as a generator of bytes """
    headers: Dict[str, str]
    """ Associated headers """


def encode(content, encoding: Optional[str] = "json") -> Response:
    """Encode content in given encoding.

    Warning: Not all encodings supports all types of content.

    :param content: Content to encode
    :param encoding:
        - `json` (default)
        - `npy`: Only nD array-like of numbers is supported
    :returns: A Response object containing content and headers
    :raises ValueError: If encoding is not "json" nor "npy"
    """
    if encoding in ("json", None):
        return Response(
            orjson_encode(content),
            headers={"Content-Type": "application/json"},
        )

    if encoding == "npy":
        return Response(
            npy_stream(content),
            headers={
                "Content-Type": "application/octet-stream",
                "Content-Disposition": 'attachment; filename="data.npy"',
            },
        )

    if encoding == "bin":
        return Response(
            bin_encode(content),
            headers={
                "Content-Type": "application/octet-stream",
            },
        )

    raise ValueError(f"Unsupported encoding {encoding}")
