#!/usr/bin/env python
# coding: utf-8
"""Flask-based server sample code"""
import argparse
from flask import Flask
from flask_compress import Compress  # type: ignore
from flask_cors import CORS  # type: ignore
import os
from h5grove.flaskutils import BLUEPRINT as h5grove_blueprint


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument(
    "-p", "--port", type=int, default=8888, help="Port the server is listening on"
)
parser.add_argument("--ip", default="localhost", help="IP the server is listening on")
parser.add_argument(
    "--compression", action="store_true", help="Enable HTTP compression"
)
parser.add_argument(
    "--basedir", default=".", help="Base directory from which to retrieve HDF5 files"
)
options = parser.parse_args()

# Create Flask application
app = Flask(__name__)

# Enable cross-origin resource sharing, see https://flask-cors.readthedocs.io
CORS(app)

# Enable compression, see https://github.com/colour-science/flask-compress
if options.compression:
    Compress(app)

# Configure h5grove default endpoints
app.config["H5_BASE_DIR"] = os.path.abspath(options.basedir)
app.register_blueprint(h5grove_blueprint)

# Start server
app.run(port=options.port, host=options.ip)
