"""Helpers for usage with `Flask <https://flask.palletsprojects.com/>`_"""

from werkzeug.exceptions import HTTPException
from flask import Blueprint, current_app, request, Response, Request
import os
from typing import Any, Callable, Mapping, Optional


from .content import (
    DatasetContent,
    ResolvedEntityContent,
    get_content_from_file,
    get_list_of_paths,
)
from .encoders import encode
from .utils import parse_bool_arg


__all__ = [
    "root_route",
    "attr_route",
    "data_route",
    "meta_route",
    "paths_route",
    "stats_route",
    "URL_RULES",
    "BLUEPRINT",
]


def make_encoded_response(
    content, format_arg: Optional[str] = "json", status: Optional[int] = None
) -> Response:
    """Prepare flask Response according to format"""
    h5grove_response = encode(content, format_arg)
    response = Response(h5grove_response.content, status=status)
    response.headers.update(h5grove_response.headers)
    return response


def get_filename(a_request: Request) -> str:
    file_path = a_request.args.get("file")
    if file_path is None:
        raise KeyError("File argument is required")

    return os.path.join(current_app.config["H5_BASE_DIR"], file_path)


def create_error(status_code: int, message: str):
    return HTTPException(
        response=make_encoded_response({"message": message}, status=status_code)
    )


def root_route():
    """`/` endpoint handler to check server status"""
    return "ok"


def attr_route():
    """`/attr/` endpoint handler"""
    filename = get_filename(request)
    path = request.args.get("path")
    attr_keys = (
        request.args.getlist("attr_keys") if "attr_keys" in request.args else None
    )

    with get_content_from_file(filename, path, create_error) as content:
        assert isinstance(content, ResolvedEntityContent)
        return make_encoded_response(content.attributes(attr_keys))


def data_route():
    """`/data/` endpoint handler"""
    filename = get_filename(request)
    path = request.args.get("path")
    selection = request.args.get("selection")
    format_arg = request.args.get("format")
    dtype = request.args.get("dtype", None)
    flatten = parse_bool_arg(request.args.get("flatten"), fallback=False)

    with get_content_from_file(filename, path, create_error) as content:
        assert isinstance(content, DatasetContent)
        data = content.data(selection, flatten, dtype)
        return make_encoded_response(data, format_arg)


def meta_route():
    """`/meta/` endpoint handler"""
    filename = get_filename(request)
    path = request.args.get("path")
    resolve_links = request.args.get("resolve_links", None)

    with get_content_from_file(filename, path, create_error, resolve_links) as content:
        return make_encoded_response(content.metadata())


def paths_route():
    filename = get_filename(request)
    path = request.args.get("path")
    resolve_links = request.args.get("resolve_links", None)

    with get_list_of_paths(filename, path, create_error, resolve_links) as paths:
        return make_encoded_response(paths)


def stats_route():
    """`/stats/` endpoint handler"""
    filename = get_filename(request)
    path = request.args.get("path")
    selection = request.args.get("selection")

    with get_content_from_file(filename, path, create_error) as content:
        assert isinstance(content, DatasetContent)
        return make_encoded_response(content.data_stats(selection))


URL_RULES = {
    "/": root_route,
    "/attr/": attr_route,
    "/data/": data_route,
    "/meta/": meta_route,
    "/paths/": paths_route,
    "/stats/": stats_route,
}
"""Mapping of Flask URL endpoints to handlers"""


BLUEPRINT = Blueprint("h5grove", __name__)
"""Blueprint of h5grove endpoints.

It relies on `H5_BASE_DIR` being defined in the app config.
"""


def _init_blueprint(blueprint: Blueprint, url_rules: Mapping[str, Callable[[], Any]]):
    for url, view_func in url_rules.items():
        blueprint.add_url_rule(url, view_func=view_func)


_init_blueprint(BLUEPRINT, URL_RULES)
