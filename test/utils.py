import io
import json
import numpy as np
from typing import List, NamedTuple, Tuple

import tifffile

from h5grove.utils import hdf_path_join


class Response(NamedTuple):
    """Return type of :meth:`get`"""

    status: int
    headers: List[Tuple[str, str]]
    content: bytes

    def find_header_value(self, key: str):
        """Find header value by key (case-insensitive)"""
        return {h[0].lower(): h[1] for h in self.headers}[key.lower()]


def test_root_path_join():
    assert hdf_path_join("/", "child") == "/child"


def test_group_path_join():
    assert hdf_path_join("/group1/group2", "data") == "/group1/group2/data"


def test_group_path_join_trailing():
    assert hdf_path_join("/group1/group2/", "data") == "/group1/group2/data"


def decode_response(response: Response, format: str = "json"):
    """Decode response content according to given format"""
    content_type = response.find_header_value("content-type")

    if format == "json":
        assert "application/json" in content_type
        return json.loads(response.content)
    if format == "npy":
        assert content_type == "application/octet-stream"
        return np.load(io.BytesIO(response.content))
    if format == "csv":
        assert content_type == "text/csv"
        return np.genfromtxt(response.content.splitlines(), delimiter=",")
    if format == "tiff":
        assert content_type == "image/tiff"
        return tifffile.imread(io.BytesIO(response.content))
    raise ValueError(f"Unsupported format: {format}")


def decode_array_response(
    response: Response,
    format: str,
    dtype: str,
    shape: Tuple[int, ...],
) -> np.ndarray:
    """Decode data array response content according to given information"""
    content_type = response.find_header_value("content-type")

    if format == "bin":
        assert content_type == "application/octet-stream"
        return np.frombuffer(response.content, dtype=dtype).reshape(shape)

    return np.asarray(decode_response(response, format))


def assert_error_response(response: Response, error_code: int):
    assert response.status == error_code
    content = decode_response(response)
    assert isinstance(content, dict) and isinstance(content["message"], str)
