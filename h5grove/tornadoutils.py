"""Helpers for usage with `Tornado <https://www.tornadoweb.org>`_"""
from h5grove.utils import NotFoundError
import os
from typing import Any, Generator, Optional
import h5py
from tornado.web import RequestHandler, MissingArgumentError, HTTPError
from .content import DatasetContent, ResolvedEntityContent, create_content
from .encoders import encode


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

    def initialize(self, base_dir, allow_origin=None) -> None:
        self.base_dir = base_dir
        self.allow_origin = allow_origin

    def get(self):
        file_path = self.get_query_argument("file")
        if file_path is None:
            raise MissingArgumentError("file")

        path = self.get_query_argument("path", None)
        format_arg = self.get_query_argument("format", None)

        full_file_path = os.path.join(self.base_dir, file_path)
        if not os.path.isfile(full_file_path):
            raise HTTPError(status_code=404, reason="File not found!")

        with h5py.File(full_file_path, "r") as h5file:
            try:
                content = self.get_content(h5file, path)
            except NotFoundError as e:
                raise HTTPError(status_code=404, reason=str(e))

        chunks, headers = encode(content, format_arg)

        for key, value in headers.items():
            self.set_header(key, value)

        if isinstance(chunks, Generator):
            for chunk in chunks:
                self.write(chunk)
        else:
            self.write(chunks)
        self.finish()

    def get_content(self, h5file, path):
        raise NotImplementedError

    def prepare(self):
        if self.allow_origin is not None:
            self.set_header("Access-Control-Allow-Origin", self.allow_origin)

    def write_error(self, status_code: int, **kwargs: Any) -> None:
        self.prepare()
        return super().write_error(status_code, **kwargs)


class AttributeHandler(BaseHandler):
    """/attr/ endpoint handler"""

    def get_content(self, h5file, path):
        content = create_content(h5file, path)
        assert isinstance(content, ResolvedEntityContent)
        return content.attributes()


class DataHandler(BaseHandler):
    """/data/ endpoint handler"""

    def get_content(self, h5file, path):
        selection = self.get_query_argument("selection", None)

        content = create_content(h5file, path)
        assert isinstance(content, DatasetContent)
        return content.data(selection)


class MetadataHandler(BaseHandler):
    """/meta/ endpoint handler"""

    def get_content(self, h5file, path):
        resolve_links_arg = self.get_query_argument("resolve_links", None)
        resolve_links = (
            resolve_links_arg.lower() != "false"
            if resolve_links_arg is not None
            else True
        )
        content = create_content(h5file, path, resolve_links)
        return content.metadata()


class StatisticsHandler(BaseHandler):
    """/stats/ endpoint handler"""

    def get_content(self, h5file, path):
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
