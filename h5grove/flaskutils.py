"""Helpers for usage with `Flask <https://flask.palletsprojects.com/>`_"""
from flask import Blueprint, current_app, request, Response
import h5py
import os
from typing import Any, Callable, Mapping, Optional

from .content import create_content, DatasetContent, ResolvedEntityContent
from .encoders import encode


__all__ = ["attr_route", "data_route", "meta_route", "URL_RULES", "BLUEPRINT"]


def make_encoded_response(content, format: Optional[str]) -> Response:
    """Prepare flask Response according to format"""
    encoded_content, headers = encode(content, format)
    response = Response(encoded_content)
    response.headers.update(headers)
    return response


def attr_route(file_path: str):
    """`/attr/<file_path>` endpoints handler"""
    filename = os.path.join(current_app.config["H5_BASE_DIR"], file_path)
    path = request.args.get("path")
    format = request.args.get("format")

    with h5py.File(filename, mode="r") as h5file:
        content = create_content(h5file, path)
        assert isinstance(content, ResolvedEntityContent)
        return make_encoded_response(content.attributes(), format)


def data_route(file_path: str):
    """`/data/<file_path>` endpoints handler"""
    filename = os.path.join(current_app.config["H5_BASE_DIR"], file_path)
    path = request.args.get("path")
    selection = request.args.get("selection")
    format = request.args.get("format")

    with h5py.File(filename, mode="r") as h5file:
        content = create_content(h5file, path)
        assert isinstance(content, DatasetContent)
        return make_encoded_response(content.data(selection), format)


def meta_route(file_path: str):
    """`/meta/<file_path>` endpoints handler"""
    filename = os.path.join(current_app.config["H5_BASE_DIR"], file_path)
    path = request.args.get("path")
    format = request.args.get("format")

    with h5py.File(filename, mode="r") as h5file:
        content = create_content(h5file, path)
        return make_encoded_response(content.metadata(), format)


def stats_route(file_path: str):
    """`/stats/<file_path>` endpoints handler"""
    filename = os.path.join(current_app.config["H5_BASE_DIR"], file_path)
    path = request.args.get("path")
    selection = request.args.get("selection")
    format = request.args.get("format")

    with h5py.File(filename, mode="r") as h5file:
        content = create_content(h5file, path)
        assert isinstance(content, DatasetContent)
        return make_encoded_response(content.data_stats(selection), format)


URL_RULES = {
    "/attr/<path:file_path>": attr_route,
    "/data/<path:file_path>": data_route,
    "/meta/<path:file_path>": meta_route,
    "/stats/<path:file_path>": stats_route,
}
"""Mapping of Flask URL endpoints to handlers"""


BLUEPRINT = Blueprint("h5grove", __name__)
"""Blueprint of h5grove endpoints.

It relies on `H5_BASE_DIR` being defined in the app config.
"""


def _init_blueprint(
    blueprint: Blueprint, url_rules: Mapping[str, Callable[[str], Any]]
):
    for url, view_func in url_rules.items():
        blueprint.add_url_rule(url, view_func=view_func)


_init_blueprint(BLUEPRINT, URL_RULES)
