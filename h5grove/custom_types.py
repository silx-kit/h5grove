from typing import TypeVar, Union
import zarr
import h5py
import numpy as np

TH5py = TypeVar("TH5py", h5py.Dataset, h5py.Datatype, h5py.Group)
TZarr = TypeVar("TZarr", zarr.Array, zarr.Group)
T = TypeVar("T", h5py.Dataset, h5py.Datatype, h5py.Group, zarr.Array, zarr.Group)
TDataset = TypeVar("TDataset", h5py.Dataset, zarr.Array)
TDatatype = TypeVar("TDatatype", h5py.Datatype, np.dtype)
TGroup = TypeVar("TGroup", h5py.Group, zarr.Group)
TFile = TypeVar("TFile", h5py.File, zarr.Group)
TAttributes = TypeVar("TAttributes", h5py.AttributeManager, zarr.core.attributes.Attributes)
TNp = TypeVar("TNp", np.ndarray, np.number, np.bool_)
