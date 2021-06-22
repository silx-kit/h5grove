"""Helpers for usage with `Tornado <https://www.tornadoweb.org>`_"""
import os
from typing import Optional
import h5py
import tornado.web
from .content import DatasetContent, ResolvedEntityContent, create_content
from .encoders import encode


__all__ = [
    "BaseHandler",
    "AttributeHandler",
    "DataHandler",
    "MetadataHandler",
    "get_handlers",
]


class BaseHandler(tornado.web.RequestHandler):
    def initialize(self, base_dir, allow_origin=None) -> None:
        self.base_dir = base_dir
        self.allow_origin = allow_origin

    def get(self, file_path):
        path = self.get_query_argument("path", None)
        format = self.get_query_argument("format", None)

        with h5py.File(os.path.join(self.base_dir, file_path), "r") as h5file:
            content = self.get_content(h5file, path)

        encoded_content_chunks, headers = encode(content, format)

        for key, value in headers.items():
            self.set_header(key, value)
        for chunk in encoded_content_chunks:
            self.write(chunk)
        self.finish()

    def get_content(self, h5file, path):
        raise NotImplementedError

    def prepare(self):
        if self.allow_origin is not None:
            self.set_header("Access-Control-Allow-Origin", self.allow_origin)


class AttributeHandler(BaseHandler):
    def get_content(self, h5file, path):
        content = create_content(h5file, path)
        assert isinstance(content, ResolvedEntityContent)
        return content.attributes()


class DataHandler(BaseHandler):
    def get_content(self, h5file, path):
        selection = self.get_query_argument("selection", None)

        content = create_content(h5file, path)
        assert isinstance(content, DatasetContent)
        return content.data(selection)


class MetadataHandler(BaseHandler):
    def get_content(self, h5file, path):
        content = create_content(h5file, path)
        return content.metadata()


class StatisticsHandler(BaseHandler):
    def get_content(self, h5file, path):
        selection = self.get_query_argument("selection", None)

        content = create_content(h5file, path)
        assert isinstance(content, DatasetContent)
        return content.data_stats(selection)


def get_handlers(base_dir: Optional[str], allow_origin: Optional[str] = None):
    """Returns list of `Rule` arguments"""
    init_args = {"base_dir": base_dir, "allow_origin": allow_origin}
    return [
        (r"/attr/(.*)", AttributeHandler, init_args),
        (r"/data/(.*)", DataHandler, init_args),
        (r"/meta/(.*)", MetadataHandler, init_args),
        (r"/stats/(.*)", StatisticsHandler, init_args),
    ]
