"""Helpers for usage with `Tornado <https://www.tornadoweb.org>`_"""
import os
from typing import Any, Optional
import h5py
from tornado.web import RequestHandler, MissingArgumentError, HTTPError

from .content import DatasetContent, ResolvedEntityContent, create_content
from .encoders import encode
from .models import LinkResolution
from .utils import NotFoundError, parse_bool_arg, parse_link_resolution_arg


__all__ = [
    "BaseHandler",
    "AttributeHandler",
    "DataHandler",
    "MetadataHandler",
    "StatisticsHandler",
    "get_handlers",
]


class BaseHandler(RequestHandler):
    """Base class for h5grove handlers"""

    def initialize(self, base_dir: str, allow_origin: Optional[str] = None) -> None:
        self.base_dir = base_dir
        self.allow_origin = allow_origin

    def get(self):
        file_path = self.get_query_argument("file")
        if file_path is None:
            raise MissingArgumentError("file")

        path = self.get_query_argument("path", None, strip=False)
        format_arg = self.get_query_argument("format", None)

        full_file_path = os.path.join(self.base_dir, file_path)
        if not os.path.isfile(full_file_path):
            raise HTTPError(status_code=404, reason="File not found!")

        with h5py.File(full_file_path, "r") as h5file:
            try:
                content = self.get_content(h5file, path)
            except NotFoundError as e:
                raise HTTPError(status_code=404, reason=str(e))

        response = encode(content, format_arg)

        for key, value in response.headers.items():
            self.set_header(key, value)

        self.write(response.content)
        self.finish()

    def get_content(self, h5file: h5py.File, path: Optional[str]):
        raise NotImplementedError

    def prepare(self):
        if self.allow_origin is not None:
            self.set_header("Access-Control-Allow-Origin", self.allow_origin)

    def write_error(self, status_code: int, **kwargs: Any) -> None:
        self.prepare()
        return super().write_error(status_code, **kwargs)


class AttributeHandler(BaseHandler):
    """/attr/ endpoint handler"""

    def get_content(self, h5file: h5py.File, path: Optional[str]):
        content = create_content(h5file, path)
        assert isinstance(content, ResolvedEntityContent)

        attr_keys = self.get_query_arguments("attr_keys", strip=False)
        # get_query_arguments returns an empty list if `attr_keys` is not present
        return content.attributes(attr_keys if len(attr_keys) > 0 else None)


class DataHandler(BaseHandler):
    """/data/ endpoint handler"""

    def get_content(self, h5file: h5py.File, path: Optional[str]):
        selection = self.get_query_argument("selection", None)
        flatten = parse_bool_arg(
            self.get_query_argument("flatten", None), fallback=False
        )

        content = create_content(h5file, path)
        assert isinstance(content, DatasetContent)
        return content.data(selection, flatten)


class MetadataHandler(BaseHandler):
    """/meta/ endpoint handler"""

    def get_content(self, h5file: h5py.File, path: Optional[str]):
        resolve_links = parse_link_resolution_arg(
            self.get_query_argument("resolve_links", None),
            fallback=LinkResolution.ONLY_VALID,
        )
        content = create_content(h5file, path, resolve_links)
        return content.metadata()


class StatisticsHandler(BaseHandler):
    """/stats/ endpoint handler"""

    def get_content(self, h5file: h5py.File, path: Optional[str]):
        selection = self.get_query_argument("selection", None)

        content = create_content(h5file, path)
        assert isinstance(content, DatasetContent)
        return content.data_stats(selection)


# TODO: Setting the return type raises mypy errors
def get_handlers(base_dir: Optional[str], allow_origin: Optional[str] = None):
    """Build h5grove handlers (`/attr`, `/data`, `/meta` and `/stats`).

    :param base_dir: Base directory from which the HDF5 files will be served
    :param allow_origin: Allowed origins for CORS
    :return type: List[Tuple[str, BaseHandler, dict]]
    """
    init_args = {"base_dir": base_dir, "allow_origin": allow_origin}
    return [
        (r"/attr/.*", AttributeHandler, init_args),
        (r"/data/.*", DataHandler, init_args),
        (r"/meta/.*", MetadataHandler, init_args),
        (r"/stats/.*", StatisticsHandler, init_args),
    ]
