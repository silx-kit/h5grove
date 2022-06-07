from fastapi import APIRouter, Depends, Response, Query, HTTPException
import h5py
import os
from typing import List, Optional

from .content import DatasetContent, ResolvedEntityContent, create_content
from .models import LinkResolution
from .utils import NotFoundError, parse_link_resolution_arg

router = APIRouter()

base_path = None


def get_content(
    h5file: h5py.File,
    path: Optional[str],
    resolve_links: LinkResolution = LinkResolution.ONLY_VALID,
):
    """Gets contents if path is in file. Raises 404 otherwise"""
    try:
        return create_content(h5file, path, resolve_links)
    except NotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))


async def add_base_path(file):
    filepath = f"{base_path}/{file}" if base_path else file
    if not os.path.isfile(filepath):
        raise HTTPException(status_code=404, detail="File not found!")
    return filepath


@router.get("/attr/")
async def get_attr(
    file: str = Depends(add_base_path),
    path: str = "/",
    attr_keys: Optional[List[str]] = Query(default=None),
):
    with h5py.File(file, "r") as h5file:
        content = get_content(h5file, path)
        assert isinstance(content, ResolvedEntityContent)
        return content.attributes(attr_keys)


@router.get("/data/")
async def get_data(
    file: str = Depends(add_base_path),
    path: str = "/",
    dtype: str = "origin",
    format: str = "json",
    flatten: bool = False,
    selection=None,
):
    with h5py.File(file, "r") as h5file:
        content = get_content(h5file, path)
        assert isinstance(content, DatasetContent)
        data = content.data(selection, flatten, dtype)
        return (
            data
            if format != "bin"
            else Response(data.tobytes(), media_type="application/octet-stream")
        )


@router.get("/meta/")
async def get_meta(
    file: str = Depends(add_base_path),
    path: str = "/",
    resolve_links: str = "only_valid",
):
    resolve_links = parse_link_resolution_arg(
        resolve_links,
        fallback=LinkResolution.ONLY_VALID,
    )
    with h5py.File(file, "r") as h5file:
        content = get_content(h5file, path, resolve_links=resolve_links)
        return content.metadata()


@router.get("/stats/")
async def get_stats(
    file: str = Depends(add_base_path), path: str = "/", selection=None
):
    with h5py.File(file, "r") as h5file:
        content = get_content(h5file, path)
        assert isinstance(content, DatasetContent)
        return content.data_stats(selection)
