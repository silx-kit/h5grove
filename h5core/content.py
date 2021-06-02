from typing import Dict, Generic, Optional, Sequence, TypeVar, Union
import h5py
import numpy as np
import os

try:
    import hdf5plugin  # noqa: F401
except ImportError:
    pass
from .utils import attr_metadata, get_entity_from_file, parse_slice, sorted_dict


class EntityContent:
    type = "other"

    def __init__(self, path: str):
        self._path = path

    def metadata(self) -> Dict[str, str]:
        return {"name": self.name, "type": self.type}

    @property
    def name(self) -> str:
        return self._path.split("/")[-1]


class ExternalLinkContent(EntityContent):
    type = "externalLink"

    def __init__(self, path: str, link: h5py.ExternalLink):
        super().__init__(path)
        self._target_file = link.filename
        self._target_path = link.path

    def metadata(self, depth=None):
        return sorted_dict(
            ("target_file", self._target_file),
            ("target_path", self._target_path),
            *super().metadata().items(),
        )


class SoftLinkContent(EntityContent):
    type = "softLink"

    def __init__(self, path: str, link: h5py.SoftLink) -> None:
        super().__init__(path)
        self._target_path = link.path

    def metadata(self, depth=None):
        return sorted_dict(
            ("target_path", self._target_path), *super().metadata().items()
        )


T = TypeVar("T", h5py.Dataset, h5py.Datatype, h5py.Group)


class ResolvedEntityContent(EntityContent, Generic[T]):
    def __init__(self, path: str, h5py_entity: T):
        super().__init__(path)
        self._h5py_entity = h5py_entity

    def attributes(self, attr_keys: Sequence[str] = None):
        if attr_keys is None:
            return dict((*self._h5py_entity.attrs.items(),))

        return dict((key, self._h5py_entity.attrs[key]) for key in attr_keys)

    def metadata(self, depth=None):
        attribute_names = sorted(self._h5py_entity.attrs.keys())
        return sorted_dict(
            (
                "attributes",
                [
                    attr_metadata(self._h5py_entity.attrs.get_id(k))
                    for k in attribute_names
                ],
            ),
            *super().metadata().items(),
        )


class DatasetContent(ResolvedEntityContent[h5py.Dataset]):
    type = "dataset"

    def metadata(self, depth=None):
        return sorted_dict(
            ("dtype", self._h5py_entity.dtype.str),
            ("shape", self._h5py_entity.shape),
            *super().metadata().items(),
        )

    def data(self, selection: str = None):
        if selection is None:
            return self._h5py_entity[()]

        parsed_slice = parse_slice(self._h5py_entity, selection)

        return self._h5py_entity[parsed_slice]

    def data_stats(
        self, selection: str = None
    ) -> Union[Dict[str, None], Dict[str, Union[float, int]]]:
        data = self._get_finite_data(selection)

        if data.size == 0:
            return {
                "min": None,
                "max": None,
                "mean": None,
                "std": None,
            }

        cast = float if np.issubdtype(data.dtype, np.floating) else int
        return {
            "min": cast(np.min(data)),
            "max": cast(np.max(data)),
            "mean": cast(np.mean(data)),
            "std": cast(np.std(data)),
        }

    def _get_finite_data(self, selection: Optional[str]) -> np.ndarray:
        data = np.array(self.data(selection), copy=False)  # So it works with scalars

        if not np.issubdtype(data.dtype, np.floating):
            return data

        mask = np.isfinite(data)
        if np.all(mask):
            return data

        return data[mask]


class GroupContent(ResolvedEntityContent[h5py.Group]):
    type = "group"

    def __init__(self, path: str, h5py_entity: h5py.Group, h5file: h5py.File):
        super().__init__(path, h5py_entity)
        self._h5file = h5file

    def _get_child_metadata_content(self, depth=0):
        return [
            create_content(self._h5file, os.path.join(self._path, child_path)).metadata(
                depth
            )
            for child_path in self._h5py_entity.keys()
        ]

    def metadata(self, depth=1):
        if depth == 0:
            return super().metadata()

        return sorted_dict(
            ("children", self._get_child_metadata_content(depth - 1)),
            *super().metadata().items(),
        )


def create_content(h5file: h5py.File, path: Optional[str]):
    if path is None:
        path = "/"

    entity = get_entity_from_file(h5file, path)

    if isinstance(entity, h5py.ExternalLink):
        return ExternalLinkContent(path, entity)

    if isinstance(entity, h5py.SoftLink):
        return SoftLinkContent(path, entity)

    if isinstance(entity, h5py.Dataset):
        return DatasetContent(path, entity)

    if isinstance(entity, h5py.Group):
        return GroupContent(path, entity, h5file)

    if isinstance(entity, h5py.Datatype):
        return ResolvedEntityContent(path, entity)

    raise TypeError(f"h5py type {type(entity)} not supported")
