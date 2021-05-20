"""Helpers for usage with `Flask <https://flask.palletsprojects.com/>`_"""
from flask import Blueprint, current_app, request, Response
import h5py
import os
from typing import Any, Callable, Dict

from .responses import create_response, DatasetResponse, ResolvedEntityResponse
from .encoders import orjson_encode


__all__ = ["attr_route", "data_route", "meta_route", "URL_RULES", "BLUEPRINT"]


def attr_route(file_path: str):
    """`/attr/<file_path>` endpoints handler"""
    filename = os.path.join(current_app.config["H5_BASE_DIR"], file_path)
    path = request.args.get("path")

    with h5py.File(filename, mode="r") as h5file:
        response = create_response(h5file, path)
        assert isinstance(response, ResolvedEntityResponse)
        return Response(
            orjson_encode(response.attributes()), mimetype="application/json"
        )


def data_route(file_path: str):
    """`/data/<file_path>` endpoints handler"""
    filename = os.path.join(current_app.config["H5_BASE_DIR"], file_path)
    path = request.args.get("path")
    selection = request.args.get("selection")

    with h5py.File(filename, mode="r") as h5file:
        response = create_response(h5file, path)
        assert isinstance(response, DatasetResponse)
        return Response(
            orjson_encode(response.data(selection)), mimetype="application/json"
        )


def meta_route(file_path: str):
    """`/meta/<file_path>` endpoints handler"""
    filename = os.path.join(current_app.config["H5_BASE_DIR"], file_path)
    path = request.args.get("path")

    with h5py.File(filename, mode="r") as h5file:
        response = create_response(h5file, path)
        return Response(orjson_encode(response.metadata()), mimetype="application/json")


URL_RULES = {
    "/attr/<path:file_path>": attr_route,
    "/data/<path:file_path>": data_route,
    "/meta/<path:file_path>": meta_route,
}
"""Mapping of Flask URL enpoints to handlers"""


BLUEPRINT = Blueprint("h5core", __name__)
"""Blueprint of h5core endpoints.

It relies on `H5_BASE_DIR` being defined in the app config.
"""


def _init_blueprint(blueprint: Blueprint, url_rules: Dict[str, Callable[[str], Any]]):
    for url, view_func in url_rules.items():
        blueprint.add_url_rule(url, view_func=view_func)


_init_blueprint(BLUEPRINT, URL_RULES)
