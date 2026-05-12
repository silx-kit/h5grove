from . import utils
from .content import create_content
from .encoders import encode
from .utils import H5FileResolver

version = "4.0.0"


def assign_resolver(resolver: H5FileResolver):
    """
    Assign a custom file resolver class.
    See the documentation of `H5FileResolver` for details.
    """
    if not isinstance(resolver, H5FileResolver):
        raise TypeError("The custom resolver must inherit from H5FileResolver.")

    utils._resolver = resolver
