# h5grove, core utilities to serve HDF5 file contents

**h5grove** is a Python package that provides utilities to design backends serving HDF5 file content: attributes, metadata and data. HDF5 files are accessed with [h5py](https://github.com/h5py/).

## Documentation

The documentation of the latest release is available [here](https://silx-kit.github.io/h5grove/).

## Rationale

There are several packages out there that can serve HDF5 files. However, they are dedicated to their usecases and settle on one implementation, hampering reusability.

In addition, some problems arise constantly when designing HDF5 backends. To name a few:

- Resolving external links
- Dealing with compression and slicing of datasets
- Encoding data efficiently and consistently (looking at you, `NaN`, `Infinity` in JSON)

**h5grove** aims at providing building blocks that solve these common problems and can be reused in existing or new backends.

## Installation

```bash
pip install h5grove
```

You can use **h5grove** low-level utilities whatever the backend implementation you choose. We simply provide additional utilities for [Tornado](https://www.tornadoweb.org/en/stable/) and [Flask](https://flask.palletsprojects.com/en/) that can be installed with:

```bash
pip install h5grove[flask] # For Flask
pip install h5grove[tornado] # For Tornado
```

## Using h5grove

### Example implementations

Example implementations using Flask and Tornado are given in the `example` folder. These are functional backends that make use of the utilities provided by the `h5grove` package. These can be run using

| Flask                          | Tornado                          |
| ------------------------------ | -------------------------------- |
| `python3 h5grove/flask_app.py` | `python3 h5grove/tornado_app.py` |

Add `-h` to have a list of the supported options.

### Integrating in an existing application

Integration in an existing Flask/Tornado application is possible thanks to the following modules:

- `flaskutils`: Utilities dedicated to Flask backends. Provides a [Blueprint](https://flask.palletsprojects.com/en/2.0.x/api/#flask.Blueprint).
- `tornadoutils`: Utilities dedicated to Tornado backends. Provides a `get_handlers` method to construct handlers that can directly passed to a Tornado application.

By adding the blueprint/handlers, you can add HDF5 serving capabilities to an existing app.

### Low-level modules

For more tailored use, you can make use of the low-level utilities in your own project.

#### Content

The key function of this module is the [create_content](https://silx-kit.github.io/h5grove/reference.html#create-a-content-object) function.

Instead of using h5py to get the desired entity using `h5file[path]`, use `create_content(h5file, path, resolve_links=True|False)` to support link resolution and dataset decompression using [hdf5plugin](https://pypi.org/project/hdf5plugin/). It will return a [Content](https://silx-kit.github.io/h5grove/reference.html#content-object-reference) object that holds handy information to design endpoints.

In addition, `Content` objects form a hierarchy of classes that expose the relevant information of the entity through methods:

- `attributes`: Only for non-link entities. The dict of attributes.
- `metadata`: For all entities. Information on the entities. Includes attribute metadata for non-link entities.
- `data`: Only for datasets. Data contained in a dataset or a slice of dataset.
- `data_stats`: Only for datasets. Statistics computed on the data of the dataset or a slice of it.

These methods are directly plugged to the endpoints from the example implementations so you can take a look at the [endpoints API]() for more information.

#### Encoders

The [encoders](https://silx-kit.github.io/h5grove/reference.html#encoders-module) module contain functions that encode data and provide the appropriate headers to build request responses. The module provides a JSON encoder using `orjson` and a binary encoder for NumPy arrays.
