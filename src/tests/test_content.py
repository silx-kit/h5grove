import h5py
import numpy as np
import pytest

from h5grove.content import get_list_of_paths
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
