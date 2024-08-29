"""Test flask_utils blueprint with Flask testing"""

from __future__ import annotations
from collections.abc import Callable

import pathlib
from flask import Flask
import pytest

from conftest import BaseServer
from utils import Response
import base_test

from h5grove.flask_utils import BLUEPRINT

# Fixtures ###


class _FlaskServer(BaseServer):
    """Class of objects provided by `flask_server` fixture"""

    def __init__(self, served_dir: pathlib.Path, client):
        super().__init__(served_dir)
        self.__client = client

    def _get_response(self, url: str, benchmark: Callable) -> Response:
        r = benchmark(lambda: self.__client.get(url))
        return Response(
            status=r.status_code, headers=list(r.headers), content=r.get_data()
        )


@pytest.fixture(scope="session")
def flask_server(tmp_path_factory):
    """Flask test client-based `server` fixture.

    Provides a function to fetch endpoints from the server.
    """
    base_dir = tmp_path_factory.mktemp("h5grove_flask_served").absolute()

    app = Flask("Test server")
    app.config["H5_BASE_DIR"] = str(base_dir)
    app.register_blueprint(BLUEPRINT)

    with app.test_client() as client:
        yield _FlaskServer(base_dir, client)


# Tests ###


class TestFlaskEndpoints(base_test.BaseTestEndpoints):
    """Test Flask blueprint endpoints using Flask test client"""

    @pytest.fixture
    def server(self, flask_server):
        yield flask_server
