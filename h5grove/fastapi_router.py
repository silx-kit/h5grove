from fastapi import APIRouter, Depends, Response
import h5py
from typing import List

from .content import DatasetContent, ResolvedEntityContent, create_content
from .models import LinkResolution
from .utils import parse_link_resolution_arg

router = APIRouter()

base_path = None


async def add_base_path(file):
    return f"{base_path}/{file}" if base_path else file


async def get_h5file(file: str = Depends(add_base_path)):
    return h5py.File(file, mode="r")


@router.get("/attr/")
async def get_attr(
    file: h5py.File = Depends(get_h5file), path: str = "/", attr_keys: List[str] = []
):
    content = create_content(file, path)
    assert isinstance(content, ResolvedEntityContent)
    return content.attributes(attr_keys)


@router.get("/data/")
async def get_data(
    file: h5py.File = Depends(get_h5file),
    path: str = "/",
    dtype: str = "origin",
    format: str = "json",
    flatten: bool = False,
    selection=None,
):
    content = create_content(file, path)
    assert isinstance(content, DatasetContent)
    data = content.data(selection, flatten, dtype)
    return (
        data
        if format != "bin"
        else Response(data.tobytes(), media_type="application/octet-stream")
    )


@router.get("/meta/")
async def get_meta(
    file: h5py.File = Depends(get_h5file),
    path: str = "/",
    resolve_links: str = "only_valid",
):
    resolve_links = parse_link_resolution_arg(
        resolve_links,
        fallback=LinkResolution.ONLY_VALID,
    )
    con = create_content(file, path)
    return con.metadata()


@router.get("/stats/")
async def get_stats(
    file: h5py.File = Depends(get_h5file), path: str = "/", selection=None
):
    content = create_content(file, path)
    assert isinstance(content, DatasetContent)
    return content.data_stats(selection)
