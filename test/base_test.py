"""Base class for testing with different servers"""
from typing import Generator
from urllib.parse import urlencode

import h5py
import numpy as np
import pytest

from conftest import BaseServer


class BaseTestEndpoints:
    """Base class for tests of different endpoints"""

    @pytest.fixture
    def server(self) -> Generator[BaseServer, None, None]:
        """Override in subclass with a fixture providing a :class:`BaseServer`"""
        raise NotImplementedError()

    def get_decoded_content(
        self, server, endpoint: str, filename: str, path: str, format: str
    ):
        """Return decoded response content of request for given endpoint and args"""
        return server.get_decoded_content(
            f"/{endpoint}/{filename}?{urlencode({'path': path, 'format': format})}",
            format=format,
        )

    def test_attr_on_root(self, server):
        """Test /attr/ endpoint on root group"""
        # Test condition
        tested_h5entity_path = "/"
        attributes = {"NX_class": "NXRoot", "default": "entry"}

        filename = "test.h5"
        with h5py.File(server.served_directory / filename, mode="w") as h5file:
            for name, value in attributes.items():
                h5file.attrs[name] = value

        retrieved_attributes = self.get_decoded_content(
            server, "attr", filename, tested_h5entity_path, format="json"
        )

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

        content = self.get_decoded_content(server, "data", filename, tested_h5entity_path, format)
        retrieved_data = np.array(content)

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

        content = self.get_decoded_content(server, "meta", filename, tested_h5entity_path, format="json")
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

        # Get from server
        retrieved_stats = self.get_decoded_content(
            server, "stats", filename, tested_h5entity_path, format="json"
        )
        assert retrieved_stats == expected_stats
