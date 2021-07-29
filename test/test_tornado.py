"""Test tornadoutils using pytest-tornado"""
import pathlib
from typing import Callable
import pytest
from tornado.httpclient import HTTPClientError
import tornado.web

from conftest import BaseServer
import base_test

from h5grove.tornadoutils import get_handlers

# Fixtures ###


@pytest.fixture(scope="session")
def _base_dir(tmp_path_factory):
    """Fixture creating base_dir.

    Needed to share base_dir between `app` and `server` fixtures
    """
    yield tmp_path_factory.mktemp("h5grove_tornado_served").absolute()


@pytest.fixture
def app(_base_dir):
    """Fixture used by pytest-tornado"""
    return tornado.web.Application(get_handlers(_base_dir), debug=True)


class _TornadoServer(BaseServer):
    """Class of objects provided by :func:`tornado_server` fixture"""

    def __init__(self, served_dir: pathlib.Path, io_loop, http_client, base_url: str):
        super().__init__(served_dir)
        self.__io_loop = io_loop
        self.__http_client = http_client
        self.__base_url = base_url

    def __fetch(self, url: str):
        """Make a synchronous fetch of given url"""
        future = self.__http_client.fetch(self.__base_url + url)
        self.__io_loop.run_sync(lambda: future)
        return future.result()

    def _get_response(self, url: str, benchmark: Callable) -> BaseServer.Response:
        r = benchmark(lambda: self.__fetch(url))
        return BaseServer.Response(
            status=r.code, headers=list(r.headers.get_all()), content=r.body
        )

    def assert_404(self, url: str):
        with pytest.raises(HTTPClientError) as e:
            self._get_response(url, lambda f: f())
        assert e.value.code == 404


@pytest.fixture
def tornado_server(_base_dir, io_loop, http_client, base_url):
    """tornado test client-based `server` fixture.

    Provides a function to fetch endpoints from the server.
    """
    yield _TornadoServer(_base_dir, io_loop, http_client, base_url)


# Tests ###


class TestTornadoEndpoints(base_test.BaseTestEndpoints):
    """Test tornado handler enpoints using pytest-tornado"""

    @pytest.fixture
    def server(self, tornado_server):
        """Override TestEndpoints.server fixture to use pytest-tornado"""
        yield tornado_server
