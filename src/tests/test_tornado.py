from __future__ import annotations

import pathlib
import threading
from collections.abc import Callable

import base_test
import httpx
import pytest
import tornado.httpserver
import tornado.ioloop
import tornado.testing
import tornado.web
from conftest import BaseServer
from utils import Response

from h5grove.tornado_utils import get_handlers

# Fixtures ###

TIMEOUT = 10


class _TornadoServer(BaseServer):
    """Class of objects provided by :func:`tornado_server` fixture"""

    def __init__(self, served_dir: pathlib.Path, base_url: str):
        super().__init__(served_dir)
        self.__client = httpx.Client(base_url=base_url)

    def _get_response(self, url: str, benchmark: Callable) -> Response:
        r = benchmark(lambda: self.__client.get(url))
        return Response(
            status=r.status_code,
            headers=[(k, v) for k, v in r.headers.items()],
            content=r.content,
        )


@pytest.fixture(scope="session")
def tornado_server(tmp_path_factory):
    """Run the tornado app in a thread and provide a `server` object."""
    base_dir = tmp_path_factory.mktemp("h5grove_tornado_served").absolute()

    io_loop = tornado.ioloop.IOLoop(make_current=False)

    sock, port = tornado.testing.bind_unused_port()
    app = tornado.web.Application(get_handlers(base_dir))
    server = tornado.httpserver.HTTPServer(app)
    # Register the listening socket in the thread of the io_loop
    io_loop.add_callback(server.add_sockets, [sock])

    thread = threading.Thread(target=io_loop.start, daemon=True)
    thread.start()
    yield _TornadoServer(base_dir, f"http://127.0.0.1:{port}")
    io_loop.add_callback(server.stop)
    io_loop.add_callback(io_loop.stop)
    thread.join(TIMEOUT)


# Tests ###


class TestTornadoEndpoints(base_test.BaseTestEndpoints):
    @pytest.fixture
    def server(self, tornado_server):
        yield tornado_server
