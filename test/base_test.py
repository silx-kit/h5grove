"""Base class for testing with different servers"""
import os
import stat
from typing import Generator
from urllib.parse import urlencode

import h5py
import numpy as np
import pytest

from conftest import BaseServer
from h5grove.models import LinkResolution
from test_utils import decode_response, decode_array_response


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
        nx_attributes = {
            "NX_class": "NXRoot",
            "default": "entry",
        }
        attributes = {
            **nx_attributes,
            "HDF_VERSION": h5py.version.hdf5_version,
        }

        filename = "test.h5"
        with h5py.File(server.served_directory / filename, mode="w") as h5file:
            for name, value in attributes.items():
                h5file.attrs[name] = value

        response = server.get(f"/attr/?file={filename}&path={tested_h5entity_path}")
        retrieved_attributes = decode_response(response)

        assert retrieved_attributes == attributes

        # Test attr_keys query parameter by getting only NX attributes
        response = server.get(
            f"/attr/?file={filename}&path={tested_h5entity_path}{''.join(f'&attr_keys={k}' for k in nx_attributes.keys())}"
        )
        retrieved_attributes = decode_response(response)
        assert retrieved_attributes == nx_attributes

    @pytest.mark.parametrize("format_arg", ("json", "npy"))
    @pytest.mark.parametrize("flatten", (False, True))
    def test_data_on_array(self, server, format_arg, flatten):
        """Test /data/ endpoint on array dataset in a group"""
        # Test condition
        tested_h5entity_path = "/entry/image"
        data = np.random.random((128, 128))

        filename = "test.h5"
        with h5py.File(server.served_directory / filename, mode="w") as h5file:
            h5file[tested_h5entity_path] = data

        response = server.get(
            f"/data/?{urlencode({'file': filename, 'path': tested_h5entity_path, 'format': format_arg, 'flatten': flatten})}"
        )
        retrieved_data = np.array(decode_response(response, format_arg))

        assert np.array_equal(retrieved_data, data.flatten() if flatten else data)

    @pytest.mark.parametrize("format_arg", ("npy", "bin"))
    @pytest.mark.parametrize("dtype_arg", ("origin", "safe"))
    def test_data_on_array_with_dtype(self, server, format_arg, dtype_arg):
        """Test /data/ endpoint on array dataset with dtype"""
        # Test condition
        tested_h5entity_path = "/entry/image"
        data = np.random.random((128, 128)).astype(">f2")
        # No Float16Array in JS => converted to float32
        ref_dtype = "<f4" if dtype_arg == "safe" else ">f2"

        filename = "test.h5"
        with h5py.File(server.served_directory / filename, mode="w") as h5file:
            h5file[tested_h5entity_path] = data

        response = server.get(
            f"/data/?{urlencode({ 'file': filename, 'path': tested_h5entity_path, 'format': format_arg, 'dtype': dtype_arg })}"
        )

        retrieved_data = decode_array_response(
            response, format_arg, ref_dtype, data.shape
        )

        assert np.array_equal(retrieved_data, data)

    @pytest.mark.parametrize("format_arg", ("json", "npy"))
    @pytest.mark.parametrize("flatten", (False, True))
    def test_data_on_slice(self, server, format_arg, flatten):
        """Test /data/ endpoint on array dataset in a group"""
        # Test condition
        tested_h5entity_path = "/entry/image"
        data = np.random.random((128, 128))

        filename = "test.h5"
        with h5py.File(server.served_directory / filename, mode="w") as h5file:
            h5file[tested_h5entity_path] = data

        response = server.get(
            f"/data/?{urlencode({'file': filename, 'path': tested_h5entity_path, 'selection': '100,0', 'format': format_arg, 'flatten': flatten})}"
        )
        retrieved_data = np.array(decode_response(response, format_arg))

        assert retrieved_data - data[100, 0] < 1e-8

    def test_meta_on_chunked_compressed_dataset(self, server):
        """Test /meta/ endpoint on a chunked and compressed dataset"""
        filename = "test.h5"
        tested_h5entity_path = "/data"

        with h5py.File(server.served_directory / filename, mode="w") as h5file:
            h5file.create_dataset(
                tested_h5entity_path,
                data=np.arange(100).reshape(10, 10),
                compression="gzip",
                shuffle=True,
                dtype="<f8",
                chunks=(5, 5),
            )

        response = server.get(f"/meta/?file={filename}&path={tested_h5entity_path}")
        content = decode_response(response)

        assert content == {
            "attributes": [],
            "filters": [{"id": 2, "name": "shuffle"}, {"id": 1, "name": "deflate"}],
            "chunks": [5, 5],
            "name": "data",
            "dtype": "<f8",
            "shape": [10, 10],
            "type": "dataset",
        }

    def test_meta_on_compound_dataset(self, server):
        """Test /meta/ endpoint on a chunked and compressed dataset"""
        filename = "test.h5"
        tested_h5entity_path = "/dogs"

        with h5py.File(server.served_directory / filename, mode="w") as h5file:
            h5file.create_dataset(
                tested_h5entity_path,
                data=np.array(
                    [("Rex", 9, 81.0), ("Fido", 3, 27.0)],
                    dtype=[("name", "S10"), ("age", "i4"), ("weight", "f4")],
                ),
            )

        response = server.get(f"/meta/?file={filename}&path={tested_h5entity_path}")
        content = decode_response(response)

        assert content == {
            "attributes": [],
            "chunks": None,
            "dtype": {"name": "|S10", "age": "<i4", "weight": "<f4"},
            "filters": None,
            "name": "dogs",
            "shape": [2],
            "type": "dataset",
        }

    def test_meta_on_group(self, server):
        """Test /meta/ endpoint on a group"""
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

        response = server.get(f"/meta/?file={filename}&path={tested_h5entity_path}")
        content = decode_response(response)
        retrieved_attr_name = [attr["name"] for attr in content["attributes"]]
        retrieved_children_name = [child["name"] for child in content["children"]]

        assert retrieved_attr_name == list(attributes.keys())
        assert retrieved_children_name == children

    @pytest.mark.parametrize(
        "resolve_links",
        (LinkResolution.NONE, LinkResolution.ONLY_VALID, LinkResolution.ALL),
    )
    def test_meta_on_valid_ext_link(self, server, resolve_links):
        source_file = "source.h5"
        with h5py.File(server.served_directory / "source.h5", mode="w") as h5file:
            data = np.arange(10, dtype="<f8")
            h5file.create_dataset("data", data=data)

        filename = "link.h5"
        with h5py.File(server.served_directory / "link.h5", mode="w") as h5file:
            h5file["ext_link"] = h5py.ExternalLink(source_file, "data")

        response = server.get(
            f"/meta/?{urlencode({'file': filename, 'path': '/ext_link', 'resolve_links': f'{resolve_links}'})}"
        )
        content = decode_response(response)

        # Valid link is not resolved only if link resolution is 'none'.
        if resolve_links == LinkResolution.NONE:
            assert content == {
                "name": "ext_link",
                "target_file": "source.h5",
                "target_path": "data",
                "type": "external_link",
            }
        else:
            assert content == {
                "attributes": [],
                "chunks": None,
                "dtype": "<f8",
                "filters": None,
                "name": "ext_link",
                "shape": [10],
                "type": "dataset",
            }

    def test_stats_on_negative_scalar(self, server):
        tested_h5entity_path = "/entry/data"
        h5data = -10
        expected_stats = {
            "strict_positive_min": None,
            "positive_min": None,
            "min": -10,
            "max": -10,
            "mean": -10,
            "std": 0,
        }

        filename = "test.h5"
        with h5py.File(server.served_directory / filename, mode="w") as h5file:
            h5file[tested_h5entity_path] = h5data

        response = server.get(f"/stats/?file={filename}&path={tested_h5entity_path}")
        retrieved_stats = decode_response(response)
        assert retrieved_stats == expected_stats

    def test_stats_on_empty_array(self, server):
        tested_h5entity_path = "/entry/data"
        h5data = []
        expected_stats = {
            "strict_positive_min": None,
            "positive_min": None,
            "min": None,
            "max": None,
            "mean": None,
            "std": None,
        }

        filename = "test.h5"
        with h5py.File(server.served_directory / filename, mode="w") as h5file:
            h5file[tested_h5entity_path] = h5data

        response = server.get(f"/stats/?file={filename}&path={tested_h5entity_path}")
        retrieved_stats = decode_response(response)
        assert retrieved_stats == expected_stats

    def test_stats_on_array(self, server):
        tested_h5entity_path = "/entry/data"
        h5data = [-10, -5, -1, 0, 1, 5, 10, np.nan, np.inf]
        expected_stats = {
            "strict_positive_min": 1,
            "positive_min": 0,
            "min": -10,
            "max": 10,
            "mean": 0.0,
            "std": 6.0,
        }

        filename = "test.h5"
        with h5py.File(server.served_directory / filename, mode="w") as h5file:
            h5file[tested_h5entity_path] = h5data

        response = server.get(f"/stats/?file={filename}&path={tested_h5entity_path}")
        retrieved_stats = decode_response(response)
        assert retrieved_stats == expected_stats

    def test_404_on_non_existing_path(self, server):
        filename = "test.h5"
        not_a_path = "not_a_path"
        with h5py.File(server.served_directory / filename, mode="w") as h5file:
            h5file["data"] = 0

        server.assert_404(f"/attr/?file={filename}&path={not_a_path}")
        server.assert_404(f"/data/?file={filename}&path={not_a_path}")
        server.assert_404(f"/meta/?file={filename}&path={not_a_path}")
        server.assert_404(f"/stats/?file={filename}&path={not_a_path}")

    def test_404_on_non_existing_file(self, server):
        filename = "not_a_file.h5"
        path = "/"

        server.assert_404(f"/attr/?file={filename}&path={path}")
        server.assert_404(f"/data/?file={filename}&path={path}")
        server.assert_404(f"/meta/?file={filename}&path={path}")
        server.assert_404(f"/stats/?file={filename}&path={path}")

    @pytest.mark.parametrize(
        "resolve_links",
        (LinkResolution.NONE, LinkResolution.ONLY_VALID, LinkResolution.ALL),
    )
    def test_meta_on_broken_soft_link(self, server, resolve_links):
        filename = "test.h5"
        link_path = "link"
        with h5py.File(server.served_directory / filename, mode="w") as h5file:
            h5file[link_path] = h5py.SoftLink("not_an_entity")

        url = f"/meta/?{urlencode({'file': filename, 'path': link_path, 'resolve_links': f'{resolve_links}'})}"

        # It should return 404 if trying to resolve the broken link
        if resolve_links == LinkResolution.ALL:
            server.assert_404(url)
        # It should return the link metadata when not resolving the link (resolve_links set to NONE or ONLY_VALID)
        else:
            response = server.get(url)
            content = decode_response(response)
            assert content == {
                "name": "link",
                "target_path": "not_an_entity",
                "type": "soft_link",
            }

    def test_403_on_file_without_read_permission(self, server):
        filename = "test.h5"
        path = "/"

        with h5py.File(server.served_directory / filename, mode="w") as h5file:
            h5file["dset"] = 10

        os.chmod(
            server.served_directory / filename,
            mode=stat.S_IWUSR,
        )

        server.assert_403(f"/attr/?file={filename}&path={path}")
        server.assert_403(f"/data/?file={filename}&path={path}")
        server.assert_403(f"/meta/?file={filename}&path={path}")
        server.assert_403(f"/stats/?file={filename}&path={path}")
