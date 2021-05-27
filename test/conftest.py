import io
import json
import os
import pathlib
import socketserver
import subprocess
import sys
import time
from typing import Callable, List, NamedTuple, Optional, Tuple
from urllib.request import urlopen

import bson
import numpy as np
import pytest


class BaseServer:
    """Base class for object provided through `server` fixture"""

    class Response(NamedTuple):
        """Return type of :meth:`get`"""

        status: int
        headers: List[Tuple[str, str]]
        content: bytes

    def __init__(self, served_dir: pathlib.Path):
        self.__served_dir = served_dir

    @property
    def served_directory(self) -> pathlib.Path:
        """Root directory served by the running server"""
        return self.__served_dir

    def _get_response(self, url: str, benchmark: Callable) -> Response:
        """Override in subclass to implement fetching response"""
        raise NotImplementedError()

    def get(self, url: str, benchmark: Optional[Callable] = None) -> Response:
        """Request url and return retrieved response"""
        if benchmark is None:
            response = self._get_response(url, lambda f: f())
        else:
            response = self._get_response(url, benchmark)

        assert response.status == 200
        content_lengths = [
            header[1] for header in response.headers if header[0] == "Content-Length"
        ]
        if content_lengths:
            assert len(response.content) == int(content_lengths[0])

        return response

    def get_decoded_content(
        self, url: str, format: str, benchmark: Optional[Callable] = None
    ):
        """Request url and return decoded content according to format"""
        response = self.get(url, benchmark)
        content_type = [h[1] for h in response.headers if h[0] == "Content-Type"][0]
        assert content_type == self._CONTENT_TYPES[format]
        return self._decode(response.content, format)

    _CONTENT_TYPES = {
        "json": "application/json",
        "bson": "application/bson",
        "npy": "application/octet-stream",
    }
    """Mapping of format: "Content-Type" header"""

    def _decode_bson(self, content):
        """Decode bson taking care of decoding bytes to utf-8 str"""
        if isinstance(content, dict):
            return dict([(k, self._decode_bson(v)) for k, v in content.items()])
        elif isinstance(content, (list, tuple)):
            return [self._decode_bson(v) for v in content]
        elif isinstance(content, bytes):
            return content.decode("utf-8")
        else:
            return content

    def _decode(self, content: bytes, format: str = "json"):
        """Decode content according to Content-Type header"""
        if format == "json":
            return json.loads(content)
        if format == "bson":
            decoded_content = self._decode_bson(bson.loads(content))
            # Handle specific storage of arrays with bson
            if tuple(decoded_content.keys()) == ("",):
                return decoded_content[""]
            return decoded_content
        if format == "npy":
            return np.load(io.BytesIO(content))
        raise ValueError(f"Unsupported format: {format}")


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


def get_free_tcp_port(host: str = "localhost") -> int:
    """Returns an available TCP port"""
    with socketserver.TCPServer((host, 0), socketserver.BaseRequestHandler) as s:
        return s.server_address[1]


@pytest.fixture(scope="module", params=("flask_app.py", "tornado_app.py"))
def subprocess_server(tmp_path_factory, request):
    """Fixture running server as a subprocess.

    Provides a function to fetch endpoints from the server.
    """
    base_dir = tmp_path_factory.mktemp("h5core_example_app_served").absolute()

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
