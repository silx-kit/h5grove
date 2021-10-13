# h5grove reference

## `content` module

### Create a content object

h5grove provides the [create_content](https://silx-kit.github.io/h5grove/reference.html#create-a-content-object) that is a h5py wrapper to get a desired h5py entity. The function handles support link resolution and dataset decompression using [hdf5plugin](https://pypi.org/project/hdf5plugin/)

```{eval-rst}
.. autofunction:: h5grove.content.create_content
```

### Content object reference

The [Content](https://silx-kit.github.io/h5grove/reference.html#content-object-reference) objects returned by [create_content](https://silx-kit.github.io/h5grove/reference.html#create-a-content-object) hold information to design endpoints.

The [Content](https://silx-kit.github.io/h5grove/reference.html#content-object-reference) objects returned by [create_content](https://silx-kit.github.io/h5grove/reference.html#create-a-content-object) expose the relevant information of the entity through methods:

- `attributes`: Only for non-link entities. The dict of attributes.
- `metadata`: For all entities. Information on the entities. Includes attribute metadata for non-link entities.
- `data`: Only for datasets. Data contained in a dataset or a slice of dataset.
- `data_stats`: Only for datasets. Statistics computed on the data of the dataset or a slice of it.

These methods are directly plugged to the endpoints from the example implementations so you can take a look at the [endpoints API](https://silx-kit.github.io/h5grove/api.html) for more information.

```{eval-rst}
.. autoclass:: h5grove.content.ExternalLinkContent
    :members:
    :inherited-members:
    :undoc-members:
.. autoclass:: h5grove.content.SoftLinkContent
    :members:
    :inherited-members:
    :undoc-members:
.. autoclass:: h5grove.content.DatasetContent
    :members:
    :inherited-members:
    :undoc-members:
.. autoclass:: h5grove.content.GroupContent
    :members:
    :inherited-members:
    :undoc-members:
```

## `encoders` module

The [encoders](https://silx-kit.github.io/h5grove/reference.html#encoders-module) module contain functions that encode data and provide the appropriate headers to build request responses. The module provides a JSON encoder using `orjson` and a binary encoder for NumPy arrays.

### General

```{eval-rst}
.. autofunction:: h5grove.encoders.encode
.. autoclass:: h5grove.encoders.Response
    :members:
```

### orjson

```{eval-rst}
.. autofunction:: h5grove.encoders.orjson_default
.. autofunction:: h5grove.encoders.orjson_encode
```

### npy

```{eval-rst}
.. autofunction:: h5grove.encoders.npy_stream
```
