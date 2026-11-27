import h5py
import numpy as np
import pytest

from h5grove.content import DatasetContent, get_list_of_paths
from h5grove.models import LinkResolution


@pytest.mark.parametrize("resolve_links", tuple(LinkResolution))
@pytest.mark.parametrize(
    ("base_path", "expected"),
    (
        (None, ["/", "/alias", "/d", "/d/loop", "/d/value", "/type"]),
        ("/", ["/", "/alias", "/d", "/d/loop", "/d/value", "/type"]),
        ("d", ["d", "d/loop", "d/value"]),
        ("/d", ["/d", "/d/loop", "/d/value"]),
    ),
)
def test_paths_do_not_open_hard_link_targets(
    tmp_path, monkeypatch, resolve_links, base_path, expected
):
    filename = tmp_path / "hard-links.h5"
    with h5py.File(filename, "w") as h5file:
        h5file["d/value"] = [1, 2]
        h5file["alias"] = h5file["d/value"]
        h5file["d/loop"] = h5file["d"]
        h5file["type"] = np.dtype("i4")

    opened = []
    getitem = h5py.Group.__getitem__

    def track_getitem(group, name):
        opened.append(name)
        return getitem(group, name)

    monkeypatch.setattr(h5py.Group, "__getitem__", track_getitem)
    with get_list_of_paths(
        filename, base_path, lambda code, message: RuntimeError(message), resolve_links
    ) as paths:
        assert paths == expected

    # Listing hard-link names should only open the starting group, regardless
    # of how many datasets, aliases or committed datatypes it contains.
    assert opened == [base_path or "/"]


def test_unfiltered_read_error_is_preserved(tmp_path, monkeypatch):
    error = OSError("Simulated I/O failure")

    def fail_read(dataset, selection):
        raise error

    with h5py.File(tmp_path / "unfiltered.h5", "w") as h5file:
        dataset = h5file.create_dataset("data", data=np.arange(4))
        content = DatasetContent("/data", dataset)
        monkeypatch.setattr("h5grove.content.get_dataset_slice", fail_read)
        with pytest.raises(OSError) as actual_error:
            content.data()
        assert actual_error.value is error


def test_checksum_error_is_not_reported_as_unavailable_filter(tmp_path):
    with h5py.File(tmp_path / "checksum.h5", "w") as h5file:
        dataset = h5file.create_dataset("data", data=np.arange(4), fletcher32=True)
        filter_mask, chunk = dataset.id.read_direct_chunk((0,))
        corrupted = bytes([chunk[0] ^ 1]) + chunk[1:]
        dataset.id.write_direct_chunk((0,), corrupted, filter_mask=filter_mask)

    with h5py.File(tmp_path / "checksum.h5", "r") as h5file:
        content = DatasetContent("/data", h5file["data"])
        with pytest.raises(OSError) as original_error:
            h5file["data"][()]
        with pytest.raises(OSError) as actual_error:
            content.data()
        assert type(actual_error.value) is type(original_error.value)
        assert str(actual_error.value) == str(original_error.value)
