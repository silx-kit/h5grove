"""Helpers for usage with `Flask <https://flask.palletsprojects.com/>`_"""
from h5grove.utils import NotFoundError
from flask import abort, Blueprint, current_app, request, Response, Request
import h5py
import os
from typing import Any, Callable, Mapping, Optional

from .content import create_content, DatasetContent, ResolvedEntityContent
from .encoders import encode


__all__ = [
    "attr_route",
    "data_route",
    "meta_route",
    "stats_route",
    "URL_RULES",
    "BLUEPRINT",
]


def make_encoded_response(content, format_arg: Optional[str]) -> Response:
    """Prepare flask Response according to format"""
    encoded_content, headers = encode(content, format_arg)
    response = Response(encoded_content)
    response.headers.update(headers)
    return response


def get_content(h5file: h5py.File, path: Optional[str], resolve_links: bool = True):
    """Gets contents if path is in file. Raises 404 otherwise"""
    try:
        return create_content(h5file, path, resolve_links)
    except NotFoundError as e:
        abort(404, str(e))


def get_filename(a_request: Request) -> str:
    file_path = a_request.args.get("file")
    if file_path is None:
        raise KeyError("File argument is required")

    full_file_path = os.path.join(current_app.config["H5_BASE_DIR"], file_path)
    if not os.path.isfile(full_file_path):
        abort(404, "File not found!")

    return full_file_path


def attr_route():
    """`/attr/` endpoint handler"""
    filename = get_filename(request)
    path = request.args.get("path")
    format_arg = request.args.get("format")

    with h5py.File(filename, mode="r") as h5file:
        content = get_content(h5file, path)
        assert isinstance(content, ResolvedEntityContent)
        return make_encoded_response(content.attributes(), format_arg)


def data_route():
    """`/data/` endpoint handler"""
    filename = get_filename(request)
    path = request.args.get("path")
    selection = request.args.get("selection")
    format_arg = request.args.get("format")

    with h5py.File(filename, mode="r") as h5file:
        content = get_content(h5file, path)
        assert isinstance(content, DatasetContent)
        return make_encoded_response(content.data(selection), format_arg)


def meta_route():
    """`/meta/` endpoint handler"""
    filename = get_filename(request)
    path = request.args.get("path")
    format_arg = request.args.get("format")
    resolve_links_arg = request.args.get("resolve_links")
    resolve_links = (
        resolve_links_arg.lower() != "false" if resolve_links_arg is not None else True
    )

    with h5py.File(filename, mode="r") as h5file:
        content = get_content(h5file, path, resolve_links)
        return make_encoded_response(content.metadata(), format_arg)


def stats_route():
    """`/stats/` endpoint handler"""
    filename = get_filename(request)
    path = request.args.get("path")
    selection = request.args.get("selection")
    format_arg = request.args.get("format")

    with h5py.File(filename, mode="r") as h5file:
        content = get_content(h5file, path)
        assert isinstance(content, DatasetContent)
        return make_encoded_response(content.data_stats(selection), format_arg)


URL_RULES = {
    "/attr/": attr_route,
    "/data/": data_route,
    "/meta/": meta_route,
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
