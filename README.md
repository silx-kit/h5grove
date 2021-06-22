# h5grove, core utilities to serve HDF5 file contents

**h5grove** is a Python package that provides utilities to design backends serving HDF5 file content: attributes, metadata and data. HDF5 files are accessed with [h5py](https://github.com/h5py/).

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

## Contents

Example implementations using Flask and Tornado are given in the `example` folder. These are functional backends that make use of the utilities provided by the `h5grove` package.

For more tailored use, you can make use of the low-level utilities in your own project. The package contains the following modules:

- `content`: A hierarchy of `Content` classes that extract the relevant information (be it attributes, metadata or data) from the file (resolving links if possible) and expose them through methods.

Ideally, getting the information from a path in the file should be as simple as:

```python
with h5py.File(filepath, "r") as h5file:
    content = create_content(h5file, path)
    # Get metadata (valid for all entities)
    content.metadata()
    # Get data (only valid for datasets)
    content.data()
```

- `encoders`: Functions that encode data and provide the appropriate headers to build request responses. The module provides a JSON and a binary encoder.
- `flaskutils`: Utilities dedicated to Flask backends. Notably provides a [Blueprint](https://flask.palletsprojects.com/en/2.0.x/api/#flask.Blueprint).
- `tornadoutils`: Utilities dedicated to Tornado backends.
