"""Helpers for usage with `FastAPI <https://fastapi.tiangolo.com/>`_"""

from fastapi import APIRouter, Depends, Response, Query, Request
from fastapi.routing import APIRoute
from pydantic_settings import BaseSettings
from typing import List, Optional, Union, Callable

from .content import (
    DatasetContent,
    ResolvedEntityContent,
    get_content_from_file,
    get_list_of_paths,
)
from .encoders import encode

__all__ = [
    "router",
    "settings",
    "get_root",
    "get_attr",
    "get_data",
    "get_meta",
    "get_stats",
]


class H5GroveRoute(APIRoute):
    def get_route_handler(self) -> Callable:
        original_route_handler = super().get_route_handler()

        async def custom_route_handler(request: Request) -> Response:
            try:
                return await original_route_handler(request)
            except H5GroveException as exc:
                return await h5grove_exception_handler(request, exc)

        return custom_route_handler


router = APIRouter(route_class=H5GroveRoute)
"""
FastAPI router with h5grove endpoints.

The directory from which files are served can be defined in `settings`.
"""


class Settings(BaseSettings):
    base_dir: Union[str, None] = None


settings = Settings()
""" Settings where base_dir can be defined """


class H5GroveException(Exception):
    def __init__(self, status_code: int, message: str) -> None:
        self.status_code = status_code
        self.message = message


def create_error(status_code, message):
    return H5GroveException(status_code, message)


async def h5grove_exception_handler(request: Request, exc: H5GroveException):
    h5grove_response = encode({"message": exc.message}, "json")
    return Response(
        content=h5grove_response.content,
        headers=h5grove_response.headers,
        status_code=exc.status_code,
    )


async def add_base_path(file):
    return f"{settings.base_dir}/{file}" if settings.base_dir else file


@router.api_route("/", methods=["GET", "HEAD"])
async def get_root():
    """`/` endpoint handler to check server status"""
    return Response("ok")


@router.get("/attr/")
async def get_attr(
    file: str = Depends(add_base_path),
    path: str = "/",
    attr_keys: Optional[List[str]] = Query(default=None),
):
    """`/attr/` endpoint handler"""
    with get_content_from_file(file, path, create_error) as content:
        assert isinstance(content, ResolvedEntityContent)
        h5grove_response = encode(content.attributes(attr_keys), "json")
        return Response(
            content=h5grove_response.content, headers=h5grove_response.headers
        )


@router.get("/data/")
async def get_data(
    file: str = Depends(add_base_path),
    path: str = "/",
    dtype: str = "origin",
    format: str = "json",
    flatten: bool = False,
    selection=None,
):
    """`/data/` endpoint handler"""
    with get_content_from_file(file, path, create_error) as content:
        assert isinstance(content, DatasetContent)
        data = content.data(selection, flatten, dtype)
        h5grove_response = encode(data, format)
        return Response(
            content=h5grove_response.content, headers=h5grove_response.headers
        )


@router.get("/meta/")
async def get_meta(
    file: str = Depends(add_base_path),
    path: str = "/",
    resolve_links: str = "only_valid",
):
    """`/meta/` endpoint handler"""

    with get_content_from_file(file, path, create_error, resolve_links) as content:
        h5grove_response = encode(content.metadata(), "json")
        return Response(
            content=h5grove_response.content, headers=h5grove_response.headers
        )


@router.get("/stats/")
async def get_stats(
    file: str = Depends(add_base_path), path: str = "/", selection=None
):
    """`/stats/` endpoint handler"""
    with get_content_from_file(file, path, create_error) as content:
        assert isinstance(content, DatasetContent)
        h5grove_response = encode(content.data_stats(selection), "json")
        return Response(
            content=h5grove_response.content, headers=h5grove_response.headers
        )


@router.get("/paths/")
async def get_paths(
    file: str = Depends(add_base_path),
    path: str = "/",
    resolve_links: str = "only_valid",
):
    with get_list_of_paths(file, path, create_error, resolve_links) as paths:
        h5grove_response = encode(paths, "json")
        return Response(
            content=h5grove_response.content, headers=h5grove_response.headers
        )
