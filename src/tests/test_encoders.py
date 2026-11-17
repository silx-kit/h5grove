import numpy as np
import orjson
import pytest

from h5grove.encoders import orjson_default, orjson_encode


@pytest.mark.parametrize("default", (None, orjson_default))
@pytest.mark.parametrize(
    ("dtype", "value"),
    (("f4", 0.1), ("u8", 2**53 + 1), ("c8", 1.25 - 2.5j)),
)
def test_non_native_scalar_json(dtype, value, default):
    data = np.array(value, dtype=np.dtype(dtype).newbyteorder("S"))
    expected = orjson.dumps(
        data[()], default=orjson_default, option=orjson.OPT_SERIALIZE_NUMPY
    )
    assert orjson_encode(data, default=default) == expected


def test_native_zero_dimensional_json_is_unchanged():
    data = np.array(0.1, dtype="f4")
    expected = orjson.dumps(
        data, default=orjson_default, option=orjson.OPT_SERIALIZE_NUMPY
    )
    assert orjson_encode(data) == expected
    assert orjson_encode(data, default=orjson_default) == expected

    received = []

    def custom_default(value):
        received.append(value)
        return "custom"

    assert orjson_encode(data, default=custom_default) == b'"custom"'
    assert received[0] is data
