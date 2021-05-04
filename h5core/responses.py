from typing import Generic, Sequence, TypeVar
import h5py
import os
from .models import EntityMetadata

try:
    import hdf5plugin  # noqa: F401
except ImportError:
    pass
from .utils import attrMetaDict, get_entity_from_file, sorted_dict


class EntityResponse:
    type = "other"

    def __init__(self, path: str):
        self._path = path

    def metadata(self) -> EntityMetadata:
        return {"name": self.name, "type": self.type}

    @property
    def name(self) -> str:
        return self._path.split("/")[-1]


class ExternalLinkResponse(EntityResponse):
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


class SoftLinkResponse(EntityResponse):
    type = "softLink"

    def __init__(self, path: str, link: h5py.SoftLink) -> None:
        super().__init__(path)
        self._target_path = link.path

    def metadata(self, depth=None):
        return sorted_dict(
            ("target_path", self._target_path), *super().metadata().items()
        )


T = TypeVar("T", h5py.Dataset, h5py.Datatype, h5py.Group)


class ResolvedEntityResponse(EntityResponse, Generic[T]):
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
                    attrMetaDict(self._h5py_entity.attrs.get_id(k))
                    for k in attribute_names
                ],
            ),
            *super().metadata().items(),
        )


class DatasetResponse(ResolvedEntityResponse[h5py.Dataset]):
    type = "dataset"

    def metadata(self, depth=None):
        return sorted_dict(
            ("dtype", self._h5py_entity.dtype.str),
            ("shape", self._h5py_entity.shape),
            *super().metadata().items(),
        )

    def data(self):
        return self._h5py_entity[()]


class GroupResponse(ResolvedEntityResponse[h5py.Group]):
    type = "group"

    def __init__(self, path: str, h5py_entity: h5py.Group, h5file: h5py.File):
        super().__init__(path, h5py_entity)
        self._h5file = h5file

    def _get_child_metadata_response(self, depth=0):
        return [
            create_response(
                self._h5file, os.path.join(self._path, child_path)
            ).metadata(depth)
            for child_path in self._h5py_entity.keys()
        ]

    def metadata(self, depth=1):
        if depth == 0:
            return super().metadata()

        return sorted_dict(
            ("children", self._get_child_metadata_response(depth - 1)),
            *super().metadata().items(),
        )


def create_response(h5file: h5py.File, path: str):
    entity = get_entity_from_file(h5file, path)

    if isinstance(entity, h5py.ExternalLink):
        return ExternalLinkResponse(path, entity)

    if isinstance(entity, h5py.SoftLink):
        return SoftLinkResponse(path, entity)

    if isinstance(entity, h5py.Dataset):
        return DatasetResponse(path, entity)

    if isinstance(entity, h5py.Group):
        return GroupResponse(path, entity, h5file)

    if isinstance(entity, h5py.Datatype):
        return ResolvedEntityResponse(path, entity)

    raise TypeError(f"h5py type {type(entity)} not supported")
