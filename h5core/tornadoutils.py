"""Helpers for usage with `Tornado <https://www.tornadoweb.org>`_"""
import os
from typing import Optional
import h5py
import tornado.web
from .responses import DatasetResponse, ResolvedEntityResponse, create_response
from .encoders import encode


__all__ = [
    "BaseHandler",
    "AttributeHandler",
    "DataHandler",
    "MetadataHandler",
    "get_handlers",
]


class BaseHandler(tornado.web.RequestHandler):
    def initialize(self, base_dir) -> None:
        self.base_dir = base_dir

    def get(self, file_path):
        path = self.get_query_argument("path", None)
        format = self.get_query_argument("format", None)

        with h5py.File(os.path.join(self.base_dir, file_path), "r") as h5file:
            response = self.get_response(h5file, path)

        encoded_content_chunks, headers = encode(response, format)

        for key, value in headers.items():
            self.set_header(key, value)
        for chunk in encoded_content_chunks:
            self.write(chunk)
        self.finish()

    def get_response(self, h5file, path):
        raise NotImplementedError


class AttributeHandler(BaseHandler):
    def get_response(self, h5file, path):
        response = create_response(h5file, path)
        assert isinstance(response, ResolvedEntityResponse)
        return response.attributes()


class DataHandler(BaseHandler):
    def get_response(self, h5file, path):
        selection = self.get_query_argument("selection", None)

        response = create_response(h5file, path)
        assert isinstance(response, DatasetResponse)
        return response.data(selection)


class MetadataHandler(BaseHandler):
    def get_response(self, h5file, path):
        response = create_response(h5file, path)
        return response.metadata()


def get_handlers(base_dir: Optional[str]):
    """Returns list of `Rule` arguments"""
    return [
        (r"/attr/(.*)", AttributeHandler, {"base_dir": base_dir}),
        (r"/data/(.*)", DataHandler, {"base_dir": base_dir}),
        (r"/meta/(.*)", MetadataHandler, {"base_dir": base_dir}),
    ]