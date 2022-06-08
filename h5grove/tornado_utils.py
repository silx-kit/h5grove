"""Helpers for usage with `Tornado <https://www.tornadoweb.org>`_"""
import os
from typing import Any, Optional
import h5py
from tornado.web import RequestHandler, MissingArgumentError, HTTPError

from .content import DatasetContent, ResolvedEntityContent, create_content
from .encoders import encode, Response
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

        full_file_path = os.path.join(self.base_dir, file_path)
        if not os.path.isfile(full_file_path):
            raise HTTPError(status_code=404, reason="File not found!")

        with h5py.File(full_file_path, "r") as h5file:
            try:
                response = self.get_response(h5file, path)
            except NotFoundError as e:
                raise HTTPError(status_code=404, reason=str(e))

        for key, value in response.headers.items():
            self.set_header(key, value)

        self.write(response.content)
        self.finish()

    def get_response(self, h5file: h5py.File, path: Optional[str]) -> Response:
        raise NotImplementedError

    def prepare(self):
        if self.allow_origin is not None:
            self.set_header("Access-Control-Allow-Origin", self.allow_origin)

    def write_error(self, status_code: int, **kwargs: Any) -> None:
        self.prepare()
        return super().write_error(status_code, **kwargs)


class AttributeHandler(BaseHandler):
    """/attr/ endpoint handler"""

    def get_response(self, h5file: h5py.File, path: Optional[str]) -> Response:
        content = create_content(h5file, path)
        assert isinstance(content, ResolvedEntityContent)

        attr_keys = self.get_query_arguments("attr_keys", strip=False)
        # get_query_arguments returns an empty list if `attr_keys` is not present
        return encode(content.attributes(attr_keys if len(attr_keys) > 0 else None))


class DataHandler(BaseHandler):
    """/data/ endpoint handler"""

    def get_response(self, h5file: h5py.File, path: Optional[str]) -> Response:
        dtype = self.get_query_argument("dtype", None)
        format_arg = self.get_query_argument("format", None)
        selection = self.get_query_argument("selection", None)
        flatten = parse_bool_arg(
            self.get_query_argument("flatten", None), fallback=False
        )

        content = create_content(h5file, path)
        assert isinstance(content, DatasetContent)
        data = content.data(selection, flatten, dtype)
        return encode(data, format_arg)


class MetadataHandler(BaseHandler):
    """/meta/ endpoint handler"""

    def get_response(self, h5file: h5py.File, path: Optional[str]) -> Response:
        resolve_links = parse_link_resolution_arg(
            self.get_query_argument("resolve_links", None),
            fallback=LinkResolution.ONLY_VALID,
        )
        content = create_content(h5file, path, resolve_links)
        return encode(content.metadata())


class StatisticsHandler(BaseHandler):
    """/stats/ endpoint handler"""

    def get_response(self, h5file: h5py.File, path: Optional[str]) -> Response:
        selection = self.get_query_argument("selection", None)

        content = create_content(h5file, path)
        assert isinstance(content, DatasetContent)
        return encode(content.data_stats(selection))


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
