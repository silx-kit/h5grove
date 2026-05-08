# Integration in an existing application

h5grove endpoints (see [API here](api.md)) can be added to existing FastAPI, Flask or Tornado applications.

## FastAPI

The module `fastapi_utils` contains utilities dedicated to FastAPI backends. It provides a [APIRouter](https://fastapi.tiangolo.com/tutorial/bigger-applications/#apirouter) that can be included in an existing FastAPI application:

```python
from fastapi import FastAPI
from h5grove.fastapi_utils import router, settings

app = FastAPI()

...

# Configure base directory from which the HDF5 files will be served
settings.base_dir = os.path.abspath(options.basedir)

app.include_router(router)
```

### fastapi_utils reference

```{eval-rst}
.. automodule:: h5grove.fastapi_utils
    :members:
```

## Flask

The module `flask_utils` contains utilities dedicated to Flask backends. It provides a [Blueprint](https://flask.palletsprojects.com/en/2.0.x/blueprints/) that can be registered to an existing Flask application:

```python
from flask import Flask
from h5grove.flask_utils import BLUEPRINT

app = Flask(__name__)

...

# Configure base directory from which the HDF5 files will be served
app.config["H5_BASE_DIR"] = os.path.abspath(options.basedir)

app.register_blueprint(BLUEPRINT)
```

### flask_utils reference

```{eval-rst}
.. automodule:: h5grove.flask_utils
    :members:
```

## Tornado

The module `tornado_utils` contains utilities dedicated to Tornado backends. It provides a `get_handlers` function to construct handlers that can be passed directly to a Tornado application:

```python
import tornado.web
from h5grove.tornado_utils import get_handlers

# The base directory from which the HDF5 files will be served is passed as argument of the function
h5grove_handlers = get_handlers(base_dir, allow_origin="*")

# On construction
app = tornado.web.Application(h5grove_handlers)

# Or using `add_handlers`
app.add_handlers(h5grove_handlers)
```

### tornado_utils reference

```{eval-rst}
.. automodule:: h5grove.tornado_utils
    :members:
```

## Custom File Resolving

By default, the HDF5 files are assumed to be on the local file system.
It is possible to serve HDF5 files stored on a remote location.
To do so, one shall subclass the `utils.H5FileResolver` class and implement the context manager protocol.
The custom resolver can then be assigned via the `assign_resolver` function.

```python
from s3fs import S3FileSystem
from h5grove import H5FileResolver, assign_resolver


class S3Resolver(H5FileResolver):
    def __init__(self, nominal_path: str):
        super().__init__(nominal_path)
        # using a local deployment for illustration
        # assuming anonymous access
        self._s3 = S3FileSystem(anon=True, endpoint_url='http://localhost:8333')
        self._fo = None

    def __enter__(self):
        self._fo = self._s3.open(self.nominal_path)
        return self._fo

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self._fo is not None:
            self._fo.close()


assign_resolver(S3Resolver)
```

The `fsspec` library allows files stored in various sources to be accessed.

However, please note that if a file-like object is returned by the resolver, the external links may not work as
intended.
Since HDF5 library has no idea about different abstractions of file systems, it is almost certain that links have to be
**manually** resolved.
