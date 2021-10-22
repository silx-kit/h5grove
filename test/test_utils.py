from h5grove.utils import hdf_path_join


def test_root_path_join():
    assert hdf_path_join("/", "child") == "/child"


def test_group_path_join():
    assert hdf_path_join("/group1/group2", "data") == "/group1/group2/data"


def test_group_path_join_trailing():
    assert hdf_path_join("/group1/group2/", "data") == "/group1/group2/data"
