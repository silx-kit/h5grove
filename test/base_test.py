"""Base class for testing with different servers"""
import io
from typing import Generator
from urllib.parse import urlencode

import h5py
import json
import numpy as np
import pytest

from conftest import BaseServer


def decode_response(response: BaseServer.Response, format: str = "json"):
    """Decode response content according to given format"""
    content_type = [h[1] for h in response.headers if h[0] == "Content-Type"][0]

    if format == "json":
        assert content_type == "application/json"
        return json.loads(response.content)
    if format == "npy":
        assert content_type == "application/octet-stream"
        return np.load(io.BytesIO(response.content))
    raise ValueError(f"Unsupported format: {format}")


class BaseTestEndpoints:
    """Base class for tests of different endpoints"""

    @pytest.fixture
    def server(self) -> Generator[BaseServer, None, None]:
        """Override in subclass with a fixture providing a :class:`BaseServer`"""
        raise NotImplementedError()

    def test_attr_on_root(self, server):
        """Test /attr/ endpoint on root group"""
        # Test condition
        tested_h5entity_path = "/"
        attributes = {"NX_class": "NXRoot", "default": "entry"}

        filename = "test.h5"
        with h5py.File(server.served_directory / filename, mode="w") as h5file:
            for name, value in attributes.items():
                h5file.attrs[name] = value

        response = server.get(f"/attr/{filename}?path={tested_h5entity_path}")
        retrieved_attributes = decode_response(response)

        assert retrieved_attributes == attributes

    @pytest.mark.parametrize("format", ("json", "npy"))
    def test_data_on_array(self, server, format):
        """Test /data/ endpoint on array dataset in a group"""
        # Test condition
        tested_h5entity_path = "/entry/image"
        data = np.random.random((128, 128))

        filename = "test.h5"
        with h5py.File(server.served_directory / filename, mode="w") as h5file:
            h5file[tested_h5entity_path] = data

        response = server.get(
            f"/data/{filename}?{urlencode({'path': tested_h5entity_path, 'format': format})}"
        )
        retrieved_data = np.array(decode_response(response, format))

        assert np.array_equal(retrieved_data, data)

    def test_meta_on_group(self, server):
        """Test /meta/ enpoint on a group"""
        # Test condition
        tested_h5entity_path = "/"
        attributes = {"NX_class": "NXRoot", "default": "entry"}
        children = ["data", "group"]

        filename = "test.h5"
        with h5py.File(server.served_directory / filename, mode="w") as h5file:
            for name, value in attributes.items():
                h5file.attrs[name] = value
            h5file.create_group("group")
            h5file["data"] = np.arange(10)

        response = server.get(f"/meta/{filename}?path={tested_h5entity_path}")
        content = decode_response(response)
        retrieved_attr_name = [attr["name"] for attr in content["attributes"]]
        retrieved_children_name = [child["name"] for child in content["children"]]

        assert retrieved_attr_name == list(attributes.keys())
        assert retrieved_children_name == children

    def test_stats_on_array(self, server):
        """Test /stats/ endpoint on an array"""
        # Test condition
        tested_h5entity_path = "/entry/image"
        image = np.random.random((128, 128))
        image[0, 0] = np.nan
        image[0, 1] = np.inf
        finite_h5data = image[np.isfinite(image)]
        expected_stats = {
            "min": np.min(finite_h5data),
            "max": np.max(finite_h5data),
            "mean": np.mean(finite_h5data),
            "std": np.std(finite_h5data),
        }

        filename = "test.h5"
        with h5py.File(server.served_directory / filename, mode="w") as h5file:
            h5file[tested_h5entity_path] = image

        response = server.get(f"/stats/{filename}?path={tested_h5entity_path}")
        retrieved_stats = decode_response(response)
        assert retrieved_stats == expected_stats
