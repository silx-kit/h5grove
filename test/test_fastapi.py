"""Test fastapi_utils with fastapi testing"""
import pathlib
from typing import Callable
from fastapi import FastAPI
from fastapi.testclient import TestClient
import pytest

from conftest import BaseServer
import base_test
from utils import Response

from h5grove.fastapi_utils import router, settings

# Fixtures ###


class _FastApiServer(BaseServer):
    def __init__(self, served_dir: pathlib.Path, client):
        super().__init__(served_dir)
        self.__client = client

    def _get_response(self, url: str, benchmark: Callable) -> Response:
        r = benchmark(lambda: self.__client.get(url))
        return Response(
            status=r.status_code,
            headers=r.headers.items(),
            content=r.content,
        )


@pytest.fixture(scope="session")
def fastapi_server(tmp_path_factory):
    """FastAPI test client-based `server` fixture.

    Provides a function to fetch endpoints from the server.
    """
    base_dir = tmp_path_factory.mktemp("h5grove_fastapi_served").absolute()

    app = FastAPI()
    settings.base_dir = str(base_dir)
    app.include_router(router)

    with TestClient(app) as client:
        yield _FastApiServer(base_dir, client)


# Tests ###


class TestFastApiEndpoints(base_test.BaseTestEndpoints):
    @pytest.fixture
    def server(self, fastapi_server):
        yield fastapi_server
