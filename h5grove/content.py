from __future__ import annotations
from collections.abc import Callable, Sequence
from typing import (
    Any,
    Generic,
    cast,
)
import os
import contextlib
from pathlib import Path
import zarr
import h5py
import numpy as np

try:
    import hdf5plugin  # noqa: F401
except ImportError:
    pass

from .custom_types import (
    T,
    TDataset,
    TGroup,
    TFile,
    TDatatype
)
from .models import (
    LinkResolution,
    Selection,
    EntityMetadata,
    ExternalLinkMetadata,
    SoftLinkMetadata,
    AttributeMetadata,
    ResolvedEntityMetadata,
    GroupMetadata,
    DatasetMetadata,
    DatatypeMetadata,
    Stats,
)
from .utils import (
    NotFoundError,
    QueryArgumentError,
    attr_metadata,
    convert,
    get_array_stats,
    open_file_with_error_fallback,
    parse_link_resolution_arg,
    get_type_metadata,
    get_filters,
    get_entity_from_file,
    hdf_path_join,
    get_dataset_slice,
    sorted_dict,
    is_h5py_file,
    is_zarr_file,
    close_file,
)


class EntityContent:
    """Base content for an entity."""

    kind = "other"

    def __init__(self, path: str):
        self._path = path

    def metadata(self) -> EntityMetadata:
        """Entity metadata"""
        return {"name": self.name, "kind": self.kind}

    @property
    def name(self) -> str:
        """Entity name (last path segment)"""
        return self._path.split("/")[-1]

    @property
    def path(self) -> str:
        """Path in the file"""
        return self._path


class ExternalLinkContent(EntityContent):
    kind = "external_link"

    def __init__(self, path: str, link: h5py.ExternalLink):
        super().__init__(path)
        self._target_file = link.filename
        self._target_path = link.path

    def metadata(self, depth=None) -> ExternalLinkMetadata:
        """External link metadata"""
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
    kind = "soft_link"

    def __init__(self, path: str, link: h5py.SoftLink) -> None:
        super().__init__(path)
        self._target_path = link.path
        """The target path of the link"""

    def metadata(self, depth=None) -> SoftLinkMetadata:
        """Soft link metadata"""
        return sorted_dict(
            ("target_path", self._target_path), *super().metadata().items()
        )

    @property
    def target_path(self) -> str:
        """The target path of the link"""
        return self._target_path


class ResolvedEntityContent(EntityContent, Generic[T]):
    """Content for a link that can be resolved into a h5py entity"""

    def __init__(self, path: str, entity: T):
        super().__init__(path)
        self._entity = entity
        """Resolved h5py or zarr entity"""

    def attributes(
        self, attr_keys: Sequence[str] | None = None
    ) -> dict[str, AttributeMetadata]:
        """Attributes of the h5py entity. Can be filtered by keys."""
        if attr_keys is None:
            return dict((*self._entity.attrs.items(),))

        return dict((key, self._entity.attrs[key]) for key in attr_keys)

    def metadata(self, depth=None) -> ResolvedEntityMetadata:
        """Resolved entity metadata"""
        attribute_names = sorted(self._entity.attrs.keys())
        return sorted_dict(
            (
                "attributes",
                [
                    attr_metadata(self._entity.attrs, name)
                    for name in attribute_names
                ],
            ),
            *super().metadata().items(),
        )

class DatasetContent(ResolvedEntityContent[TDataset]):
    kind = "dataset"

    def metadata(self, depth=None) -> DatasetMetadata:
        """Dataset metadata"""
        return sorted_dict(
            ("chunks", self._entity.chunks),
            ("filters", get_filters(self._entity)),
            ("shape", self._entity.shape),
            ("type", get_type_metadata(self._entity)),
            *super().metadata().items(),
        )

    def data(
        self,
        selection: Selection | None = None,
        flatten: bool = False,
        dtype: str | None = "origin",
    ):
        """Dataset data.

        :param selection: Slicing information
        :param flatten: True to flatten the returned array
        :param dtype: Data type conversion query parameter
          - `origin` (default): No conversion
          - `safe`: Convert to a type supported by JS typedarray (https://developer.mozilla.org/fr/docs/Web/JavaScript/Reference/Global_Objects/TypedArray)
        """
        result = convert(get_dataset_slice(self._entity, selection), dtype)

        # Do not flatten scalars nor h5py.Empty
        if flatten and isinstance(result, np.ndarray):
            return np.ravel(result)

        return result

    def data_stats(self, selection: Selection | None = None) -> Stats:
        """Statistics on the data. Providing a selection will compute stats only on the selected slice.

        :param selection: NumPy-like indexing to define a selection as a slice
        """
        data = self._get_finite_data(selection)

        return get_array_stats(data)

    def _get_finite_data(self, selection: Selection) -> np.ndarray:
        data = np.asarray(self.data(selection))  # So it works with scalars

        if not np.issubdtype(data.dtype, np.floating):
            return data

        mask = np.isfinite(data)
        if np.all(mask):
            return data

        return data[mask]


class GroupContent(ResolvedEntityContent[TGroup]):
    kind = "group"

    def __init__(self, path: str, entity: TGroup, file_entity: TFile):
        super().__init__(path, entity)
        self._file_entity = file_entity
        """File in which the entity was resolved. This is needed to resolve child entity."""

    def _get_child_metadata_content(self, depth=0):
        return [
            create_content(
                self._file_entity, os.path.join(self._path, child_path)
            ).metadata(depth)
            for child_path in self._entity.keys()
        ]

    def metadata(self, depth: int = 1) -> GroupMetadata:
        """Metadata of the group. Recursively includes child metadata if depth > 0.

        :parameter depth: The level of child metadata resolution.
        """
        if depth <= 0:
            return cast(GroupMetadata, super().metadata())

        return sorted_dict(
            ("children", self._get_child_metadata_content(depth - 1)),
            *super().metadata().items(),
        )


class DatatypeContent(ResolvedEntityContent[TDatatype]):
    kind = "datatype"

    def metadata(self, depth=None) -> DatatypeMetadata:
        """Datatype metadata"""
        return sorted_dict(
            ("type", get_type_metadata(self._entity)),
            *super().metadata().items(),
        )


def create_content(
    file_entity: TFile,
    path: str | None,
    resolve_links: LinkResolution = LinkResolution.ONLY_VALID,
):
    """
    Factory function to get entity content from a HDF5 file.
    This handles external/soft link resolution and dataset decompression.

    :param file_entity: An open HDF5 file containing the entity
    :param path: Path to the entity in the file.
    :param resolve_links: Tells which external and soft links should be resolved. Defaults to resolving only valid links.
    :raises h5grove.utils.PathError: If the path cannot be found in the file
    :raises h5grove.utils.LinkError: If a link cannot be resolved when resolve_links is set to LinkResolution.ALL.
    :raises TypeError: If encountering an unsupported h5py entity
    """
    if path is None:
        path = "/"

    entity = get_entity_from_file(file_entity, path, resolve_links)

    if isinstance(entity, h5py.ExternalLink):
        return ExternalLinkContent(path, entity)

    if isinstance(entity, h5py.SoftLink):
        return SoftLinkContent(path, entity)

    if isinstance(entity, (h5py.Dataset, zarr.Array)):
        return DatasetContent(path, entity)

    if isinstance(entity, (h5py.Group, zarr.Group)):
        return GroupContent(path, entity, file_entity)

    if isinstance(entity, (h5py.Datatype, np.dtype)):
        return DatatypeContent(path, entity)



    raise TypeError(f"Entity {type(entity)} not supported")


@contextlib.contextmanager
def get_content_from_file(
    filepath: str | Path,
    path: str | None,
    create_error: Callable[[int, str], Exception],
    resolve_links_arg: str | None = LinkResolution.ONLY_VALID,
    h5py_options: dict[str, Any] = {},
):
    f = open_file_with_error_fallback(filepath, create_error, h5py_options)

    if is_zarr_file(filepath):
        path = path.lstrip('/')

    try:
        resolve_links = parse_link_resolution_arg(
            resolve_links_arg,
            fallback=LinkResolution.ONLY_VALID,
        )
    except QueryArgumentError as e:
        close_file(filepath, f)
        raise create_error(422, str(e))
    try:
        yield create_content(f, path, resolve_links)
    except NotFoundError as e:
        raise create_error(404, str(e))
    except QueryArgumentError as e:
        raise create_error(422, str(e))
    finally:
        close_file(filepath, f)


@contextlib.contextmanager
def get_list_of_paths(
    filepath: str | Path,
    base_path: str | None,
    create_error: Callable[[int, str], Exception],
    resolve_links_arg: str | None = LinkResolution.ONLY_VALID,
    open_options: dict[str, Any] = {},
):
    f = open_file_with_error_fallback(filepath, create_error, open_options)

    try:
        resolve_links = parse_link_resolution_arg(
            resolve_links_arg,
            fallback=LinkResolution.ONLY_VALID,
        )
    except QueryArgumentError as e:
        raise create_error(422, str(e))

    names = []

    if is_h5py_file(filepath):
        def get_path(name: bytes):
            full_path = hdf_path_join(base_path, name.decode())
            content = create_content(f, full_path, resolve_links)
            names.append(content.path)


        try:
            base_content = create_content(f, base_path, resolve_links)
            assert isinstance(base_content, GroupContent)
            names.append(base_content.path)
            base_content._entity.id.links.visit(get_path)
            yield names
        except NotFoundError as e:
            raise create_error(404, str(e))
        except QueryArgumentError as e:
            raise create_error(422, str(e))
        finally:
            close_file(filepath, f)
    else:
        entity = get_entity_from_file(f, base_path, resolve_links)
        if isinstance(entity, zarr.Group):
            yield [m[0] for m in entity.members(max_depth=None)]
        else:
            yield entity.path
