import importlib.metadata

from .content import create_content
from .encoders import encode

version = importlib.metadata.version("h5grove")
