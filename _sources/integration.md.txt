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
