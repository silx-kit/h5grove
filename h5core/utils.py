import h5py
from typing import Any, Tuple
from .models import H5pyEntity


def attrMetaDict(attrId):
    return {"dtype": attrId.dtype.str, "name": attrId.name, "shape": attrId.shape}


def get_entity_from_file(h5file: h5py.File, path: str) -> H5pyEntity:
    if path == "/":
        return h5file[path]

    link = h5file.get(path, getlink=True)
    if isinstance(link, h5py.ExternalLink) or isinstance(link, h5py.SoftLink):
        try:
            return h5file[path]
        except (OSError, KeyError):
            return link

    return h5file[path]


def sorted_dict(*args: Tuple[str, Any]):
    return dict(sorted(args))
