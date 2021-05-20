import io
from numbers import Number
from typing import Generator, Sequence, Union
import numpy as np
import orjson
import h5py
from .utils import sanitize_array


def default(o) -> Union[list, str, None]:
    if isinstance(o, np.generic) or isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, complex):
        return [o.real, o.imag]
    if isinstance(o, h5py.Empty):
        return None
    if isinstance(o, bytes):
        return o.decode()
    raise TypeError


def orjson_encode(response):
    return orjson.dumps(response, default=default, option=orjson.OPT_SERIALIZE_NUMPY)


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
