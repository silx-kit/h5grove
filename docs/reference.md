# h5grove reference

## `content` module

### Create a content object

```{eval-rst}
.. autofunction:: h5grove.content.create_content
```

### Content object reference

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
