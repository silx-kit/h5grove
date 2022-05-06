from typing import Dict, Generic, Optional, Sequence, TypeVar, Union
import h5py
import numpy as np

try:
    import hdf5plugin  # noqa: F401
except ImportError:
    pass

from .models import LinkResolution, Selection
from .utils import (
    attr_metadata,
    convert,
    get_array_stats,
    get_filters,
    get_entity_from_file,
    hdf_path_join,
    get_dataset_slice,
    sorted_dict,
)


class EntityContent:
    """Base content for an entity."""

    type = "other"

    def __init__(self, path: str):
        self._path = path

    def metadata(self) -> Dict[str, str]:
        """Entity metadata

        :returns: {"name": str, "type": str}
        """
        return {"name": self.name, "type": self.type}

    @property
    def name(self) -> str:
        """Entity name. Last member of the path."""
        return self._path.split("/")[-1]

    @property
    def path(self) -> str:
        """Path in the file."""
        return self._path


class ExternalLinkContent(EntityContent):
    type = "external_link"

    def __init__(self, path: str, link: h5py.ExternalLink):
        super().__init__(path)
        self._target_file = link.filename
        self._target_path = link.path

    def metadata(self, depth=None):
        """External link metadata

        :returns: {"name": str, "target_file": str, "target_path": str, "type": str}
        """
        return sorted_dict(
            ("target_file", self._target_file),
            ("target_path", self._target_path),
            *super().metadata().items(),
        )

    @property
    def target_file(self) -> str:
        """The target file of the link"""
        return self._target_file

    @property
    def target_path(self) -> str:
        """The target path of the link (in the target file)"""
        return self._target_path


class SoftLinkContent(EntityContent):
    type = "soft_link"

    def __init__(self, path: str, link: h5py.SoftLink) -> None:
        super().__init__(path)
        self._target_path = link.path
        """ The target path of the link """

    def metadata(self, depth=None):
        """
        :returns: {"name": str, "target_path": str, "type": str}
        """
        return sorted_dict(
            ("target_path", self._target_path), *super().metadata().items()
        )

    @property
    def target_path(self) -> str:
        """The target path of the link"""
        return self._target_path


T = TypeVar("T", h5py.Dataset, h5py.Datatype, h5py.Group)


class ResolvedEntityContent(EntityContent, Generic[T]):
    """Content for a link that can be resolved into a h5py entity"""

    def __init__(self, path: str, h5py_entity: T):
        super().__init__(path)
        self._h5py_entity = h5py_entity
        """ Resolved h5py entity """

    def attributes(self, attr_keys: Optional[Sequence[str]] = None):
        """Attributes of the h5py entity. Can be filtered by keys."""
        if attr_keys is None:
            return dict((*self._h5py_entity.attrs.items(),))

        return dict((key, self._h5py_entity.attrs[key]) for key in attr_keys)

    def metadata(self, depth=None):
        """
        :returns: {"attributes": AttributeMetadata, "name": str, "type": str}
        """
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
        """
        :returns: {"attributes": AttributeMetadata, chunks": tuple, "dtype": str, "filters": tuple, "shape": tuple, "name": str, "type": str}
        """
        return sorted_dict(
            ("chunks", self._h5py_entity.chunks),
            ("dtype", self._h5py_entity.dtype.str),
            ("filters", get_filters(self._h5py_entity)),
            ("shape", self._h5py_entity.shape),
            *super().metadata().items(),
        )

    def data(
        self,
        selection: Selection = None,
        flatten: bool = False,
        dtype: Optional[str] = "origin",
    ):
        """Dataset data.

        :param selection: Slicing information
        :param flatten: True to flatten the returned array
        :param dtype: Data type conversion query parameter
          - `origin` (default): No conversion
          - `safe`: Convert to a type supported by JS typedarray (https://developer.mozilla.org/fr/docs/Web/JavaScript/Reference/Global_Objects/TypedArray)
        """
        result = convert(get_dataset_slice(self._h5py_entity, selection), dtype)

        # Do not flatten scalars nor h5py.Empty
        if flatten and isinstance(result, np.ndarray):
            return np.ravel(result)

        return result

    def data_stats(
        self, selection: Selection = None
    ) -> Dict[str, Union[float, int, None]]:
        """Statistics on the data. Providing a selection will compute stats only on the selected slice.

        :param selection: NumPy-like indexing to define a selection as a slice
        :returns: {"strict_positive_min": number | None, "positive_min": number | None, "min": number | None, "max": number | None, "mean": number | None, "std": number | None}
        """
        data = self._get_finite_data(selection)

        return get_array_stats(data)

    def _get_finite_data(self, selection: Selection) -> np.ndarray:
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
        """ File in which the entity was resolved. This is needed to resolve child entity. """

    def _get_child_metadata_content(self, depth=0):
        return [
            create_content(
                self._h5file, hdf_path_join(self._path, child_path)
            ).metadata(depth)
            for child_path in self._h5py_entity.keys()
        ]

    def metadata(self, depth: int = 1):
        """Metadata of the group. Recursively includes child metadata if depth > 0.

        :parameter depth: The level of child metadata resolution.
        :returns: {"attributes": AttributeMetadata, "children": ChildMetadata, "name": str, "type": str}
        """
        if depth <= 0:
            return super().metadata()

        return sorted_dict(
            ("children", self._get_child_metadata_content(depth - 1)),
            *super().metadata().items(),
        )


def create_content(
    h5file: h5py.File,
    path: Optional[str],
    resolve_links: LinkResolution = LinkResolution.ONLY_VALID,
):
    """
    Factory function to get entity content from a HDF5 file.
    This handles external/soft link resolution and dataset decompression.

    :param h5file: An open HDF5 file containing the entity
    :param path: Path to the entity in the file.
    :param resolve_links: Tells which external and soft links should be resolved. Defaults to resolving only valid links.
    :raises h5grove.utils.PathError: If the path cannot be found in the file
    :raises h5grove.utils.LinkError: If a link cannot be resolved when resolve_links is set to LinkResolution.ALL.
    :raises TypeError: If encountering an unsupported h5py entity
    """
    if path is None:
        path = "/"

    entity = get_entity_from_file(h5file, path, resolve_links)

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
