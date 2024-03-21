"""Helpers for usage with `Tornado <https://www.tornadoweb.org>`_"""

import os
from typing import Any, Optional

from tornado.web import HTTPError, MissingArgumentError, RequestHandler

from .content import (
    DatasetContent,
    EntityContent,
    ResolvedEntityContent,
    get_content_from_file,
    get_list_of_paths,
)
from .encoders import Response, encode
from .utils import parse_bool_arg

__all__ = [
    "RootHandler",
    "BaseHandler",
    "AttributeHandler",
    "DataHandler",
    "MetadataHandler",
    "StatisticsHandler",
    "get_handlers",
]


def create_error(status_code: int, message: str):
    return HTTPError(status_code=status_code, reason=message)


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

        response = self.get_response(
            full_file_path, path, self.get_query_argument("resolve_links", None)
        )

        for key, value in response.headers.items():
            self.set_header(key, value)

        self.write(response.content)
        self.finish()

    def get_response(
        self, full_file_path: str, path: Optional[str], resolve_links: Optional[str]
    ) -> Response:
        raise NotImplementedError

    def prepare(self):
        if self.allow_origin is not None:
            self.set_header("Access-Control-Allow-Origin", self.allow_origin)

    def write_error(self, status_code: int, **kwargs: Any) -> None:
        self.prepare()
        self.finish({"message": self._reason})


class RootHandler(BaseHandler):
    """`/` endpoint handler to check server status"""

    def get(self):
        self.finish("ok")

    def head(self):
        self.finish()


class ContentHandler(BaseHandler):
    def get_response(
        self, full_file_path: str, path: Optional[str], resolve_links: Optional[str]
    ) -> Response:
        with get_content_from_file(
            full_file_path, path, create_error, resolve_links
        ) as content:
            response = self.get_content_response(content)

        return response

    def get_content_response(self, content: EntityContent) -> Response:
        raise NotImplementedError


class AttributeHandler(ContentHandler):
    """`/attr/` endpoint handler"""

    def get_content_response(self, content: EntityContent) -> Response:
        assert isinstance(content, ResolvedEntityContent)

        attr_keys = self.get_query_arguments("attr_keys", strip=False)
        # get_query_arguments returns an empty list if `attr_keys` is not present
        return encode(content.attributes(attr_keys if len(attr_keys) > 0 else None))


class DataHandler(ContentHandler):
    """`/data/` endpoint handler"""

    def get_content_response(self, content: EntityContent) -> Response:
        dtype = self.get_query_argument("dtype", None)
        format_arg = self.get_query_argument("format", None)
        selection = self.get_query_argument("selection", None)
        flatten = parse_bool_arg(
            self.get_query_argument("flatten", None), fallback=False
        )

        assert isinstance(content, DatasetContent)
        data = content.data(selection, flatten, dtype)
        return encode(data, format_arg)


class MetadataHandler(ContentHandler):
    """`/meta/` endpoint handler"""

    def get_content_response(self, content: EntityContent) -> Response:
        return encode(content.metadata())


class StatisticsHandler(ContentHandler):
    """`/stats/` endpoint handler"""

    def get_content_response(self, content: EntityContent) -> Response:
        selection = self.get_query_argument("selection", None)

        assert isinstance(content, DatasetContent)
        return encode(content.data_stats(selection))


class PathsHandler(BaseHandler):
    def get_response(
        self, full_file_path: str, path: Optional[str], resolve_links: Optional[str]
    ) -> Response:
        with get_list_of_paths(
            full_file_path, path, create_error, resolve_links
        ) as paths:
            return encode(paths)


# TODO: Setting the return type raises mypy errors
def get_handlers(base_dir: Optional[str], allow_origin: Optional[str] = None):
    """Build h5grove handlers (`/`, `/attr/`, `/data/`, `/meta/` and `/stats/`).

    :param base_dir: Base directory from which the HDF5 files will be served
    :param allow_origin: Allowed origins for CORS
    :return type: List[Tuple[str, BaseHandler, dict]]
    """
    init_args = {"base_dir": base_dir, "allow_origin": allow_origin}
    return [
        (r"/", RootHandler, init_args),
        (r"/attr/.*", AttributeHandler, init_args),
        (r"/data/.*", DataHandler, init_args),
        (r"/meta/.*", MetadataHandler, init_args),
        (r"/paths/.*", PathsHandler, init_args),
        (r"/stats/.*", StatisticsHandler, init_args),
    ]
