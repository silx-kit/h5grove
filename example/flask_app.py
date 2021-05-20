#!/bin/env python
# coding: utf-8
"""Flask-based server sample code"""

import os

from typing import Optional

from flask import Blueprint, current_app, Flask, request, Response
from flask_cors import CORS
from flask_compress import Compress

import sys

sys.path.insert(0, ".")
from h5core.responses import create_response, DatasetResponse, ResolvedEntityResponse
from h5core.utils import stream_array_as_npy
from h5core import encoders

import bson

import h5py

try:
    import hdf5plugin  # noqa
except ImportError:
    pass

# TODO/ideas:
# Use a h5py.File opened pool? issue per user pool?
# error handling?

# DONE
# Put in common with h5py.File
# Always use same default value (None) for optional request args
# For multiple encoding add a encode(response, encoding) function or decorator?


# response functions


def attr_response(filename, path):
    with h5py.File(filename, mode="r") as h5file:
        response = create_response(h5file, path)
        assert isinstance(response, ResolvedEntityResponse)
        return response.attributes()


def data_response(filename, path, selection):
    with h5py.File(filename, mode="r") as h5file:
        response = create_response(h5file, path)
        assert isinstance(response, DatasetResponse)
        return response.data(selection)


def meta_response(filename, path):
    with h5py.File(filename, mode="r") as h5file:
        response = create_response(h5file, path)
        return response.metadata()



# flask routes


def response(payload, encoding: Optional[str] = "json"):
    if encoding is None:
        encoding = "json"

    if encoding == "json":
        return Response(encoders.orjson_encode(payload), mimetype="application/json")

    elif encoding == "bson":
        # bson only accept dict
        response = Response(
            bson.dumps(
                payload if isinstance(payload, dict) else {"": payload},
                on_unknown=encoders.default,
            ),
            mimetype="application/bson",
        )
        response.headers["Content-Disposition"] = 'attachment; filename="data.bson"'
        return response

    elif encoding == "npy":
        response = Response(
            stream_array_as_npy(payload),
            mimetype="application/octet-stream",
        )
        response.headers["Content-Disposition"] = 'attachment; filename="data.npy"'
        return response

    else:
        raise ValueError("Unsupported encoding: %s" % encoding)


def attr_route(file_path):
    return response(
        attr_response(
            filename=os.path.join(current_app.config["H5_BASE_DIR"], file_path),
            path=request.args.get("path"),
        ),
        encoding=request.args.get("format"),
    )


def data_route(file_path):
    return response(
        data_response(
            filename=os.path.join(current_app.config["H5_BASE_DIR"], file_path),
            path=request.args.get("path"),
            selection=request.args.get("selection"),
        ),
        encoding=request.args.get("format"),
    )


def meta_route(file_path):
    return response(
        meta_response(
            filename=os.path.join(current_app.config["H5_BASE_DIR"], file_path),
            path=request.args.get("path"),
        ),
        encoding=request.args.get("format"),
    )


url_rules = {
    "/attr/<path:file_path>": attr_route,
    "/data/<path:file_path>": data_route,
    "/meta/<path:file_path>": meta_route,
}
"""Mapping of Flask URL enpoints to handlers"""


blueprint = Blueprint("h5core", __name__)
"""Blueprint of h5core endpoints.

It relies on `H5_BASE_DIR` being defined in the app config.
"""

for url, view_func in url_rules.items():
    blueprint.add_url_rule(url, view_func=view_func)


def init_app(base_dir: str = ".", compression: bool = False):
    app = Flask(__name__)
    CORS(app)

    if compression:  # HTTP compression for data
        # See https://github.com/colour-science/flask-compress
        # app.config["COMPRESS_REGISTER"] = False  # disable default compression of all eligible requests
        app.config["COMPRESS_MIMETYPES"] = [
            "application/octet-stream",
            "application/bson",
        ]
        compress = Compress()
        compress.init_app(app)

    app.config["H5_BASE_DIR"] = os.path.abspath(base_dir)
    app.register_blueprint(blueprint)

    return app


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "-p", "--port", type=int, default=8888, help="Port the server is listening on"
    )
    parser.add_argument(
        "--ip", default="localhost", help="IP the server is listening on"
    )
    parser.add_argument(
        "--compression", action="store_true", help="Enable HTTP compression"
    )
    parser.add_argument(
        "--basedir",
        default=".",
        help="Base directory from which to retrieve HDF5 files",
    )
    options = parser.parse_args()

    app = init_app(base_dir=options.basedir, compression=options.compression)
    app.run(port=options.port, host=options.ip)
