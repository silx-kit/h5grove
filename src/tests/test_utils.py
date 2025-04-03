from h5grove.utils import parse_slice


def test_parse_slice():
    assert parse_slice("5") == (5,)
    assert parse_slice("1, 2:5") == (1, slice(2, 5))
    assert parse_slice("0:10:5, 2, 3:") == (slice(0, 10, 5), 2, slice(3, None))
