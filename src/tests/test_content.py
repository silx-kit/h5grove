import h5py
import numpy as np
import pytest

from h5grove.content import DatasetContent


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
