import os
import pathlib
import socketserver
import subprocess
import sys
import time
from typing import Callable, List, NamedTuple, Optional, Tuple
from urllib.request import urlopen
from urllib.error import HTTPError

import pytest


class BaseServer:
    """Base class for object provided through `server` fixture"""

    class Response(NamedTuple):
        """Return type of :meth:`get`"""

        status: int
        headers: List[Tuple[str, str]]
        content: bytes

        def find_header_value(self, key: str):
            """Find header value by key (case-insensitive)"""
            return {h[0].lower(): h[1] for h in self.headers}[key.lower()]

    def __init__(self, served_dir: pathlib.Path):
        self.__served_dir = served_dir

    @property
    def served_directory(self) -> pathlib.Path:
        """Root directory served by the running server"""
        return self.__served_dir

    def _get_response(self, url: str, benchmark: Callable) -> Response:
        """Override in subclass to implement fetching response"""
        raise NotImplementedError()

    def get(
        self,
        url: str,
        benchmark: Optional[Callable] = None,
    ) -> Response:
        """Request url and return retrieved response"""
        if benchmark is None:
            response = self._get_response(url, lambda f: f())
        else:
            response = self._get_response(url, benchmark)

        assert response.status == 200
        content_lengths = response.find_header_value("content-length")
        if content_lengths:
            assert len(response.content) == int(content_lengths)

        return response

    def assert_404(self, url: str):
        raise NotImplementedError()


# subprocess_server fixture  ###


class SubprocessServer(BaseServer):
    """Class of objects provided through subprocess-based `server` fixture"""

    def __init__(self, served_dir: pathlib.Path, base_url):
        super().__init__(served_dir)
        self.__base_url = base_url

    def _get_response(self, url: str, benchmark: Callable) -> BaseServer.Response:
        r = benchmark(lambda: urlopen(self.__base_url + url))
        return BaseServer.Response(
            status=r.status, headers=r.headers.items(), content=r.read()
        )

    def assert_404(self, url: str):
        with pytest.raises(HTTPError) as e:
            self._get_response(url, lambda f: f())
            assert e.value.code == 404


def get_free_tcp_port(host: str = "localhost") -> int:
    """Returns an available TCP port"""
    with socketserver.TCPServer((host, 0), socketserver.BaseRequestHandler) as s:
        return s.server_address[1]


@pytest.fixture(scope="module", params=("flask_app.py", "tornado_app.py"))
def subprocess_server(tmp_path_factory, request):
    """Fixture running server as a subprocess.

    Provides a function to fetch endpoints from the server.
    """
    base_dir = tmp_path_factory.mktemp("h5grove_example_app_served").absolute()

    project_root_dir = pathlib.Path(__file__).absolute().parent.parent
    host = "localhost"
    port = str(get_free_tcp_port(host))
    cmd = [
        sys.executable,
        str(pathlib.Path(project_root_dir, "example", request.param)),
        "--ip",
        host,
        "--port",
        port,
        "--basedir",
        f"{str(base_dir)}",
    ]
    env = os.environ.copy()
    env["PYTHONPATH"] = f"{str(project_root_dir)}:{env.get('PYTHONPATH', '')}"
    process = subprocess.Popen(cmd, env=env)
    time.sleep(1)
    assert process.poll() is None  # Check that server is running

    yield SubprocessServer(served_dir=base_dir, base_url=f"http://{host}:{port}")

    process.terminate()
    assert process.wait(timeout=4) is not None  # Check that server is stopped
