# Integration in an existing application

h5grove endpoints (see [API here](api.md)) can be added to existing Flask/Tornado applications.

## Flask

The module `flaskutils` contains utilities dedicated to Flask backends. It provides a [Blueprint](https://flask.palletsprojects.com/en/2.0.x/blueprints/) that can be registered to an existing Flask application:

```python
from flask import Flask
from h5grove.flaskutils import BLUEPRINT

app = Flask(__name__)

...

# Configure base directory from which the HDF5 files will be served
app.config["H5_BASE_DIR"] = os.path.abspath(options.basedir)

app.register_blueprint(BLUEPRINT)
```

### flaskutils reference

```{eval-rst}
.. automodule:: h5grove.flaskutils
    :members:
```

## Tornado

The module `tornadoutils` contains utilities dedicated to Flask backends. It provides a `get_handlers` function to construct handlers that can be passed directly to a Tornado application:

```python
import tornado.web
from h5grove.tornadoutils import get_handlers

# The base directory from which the HDF5 files will be served is passed as argument of the function
h5grove_handlers = get_handlers(base_dir, allow_origin="*")

# On construction
app = tornado.web.Application(h5grove_handlers)

# Or using `add_handlers`
app.add_handlers(h5grove_handlers)
```

### tornadoutils reference

```{eval-rst}
.. automodule:: h5grove.tornadoutils
    :members:
```
