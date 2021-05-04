from typing import Union
import numpy as np
import orjson
import h5py


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
