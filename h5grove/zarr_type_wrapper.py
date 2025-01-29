import numpy as np

from .models import TypeMetadata


class ZarrTypeWrapper():
    def __init__(self, type_id: np.dtype, shape: tuple):

        self.type_id = type_id
        self.shape = shape

        for key in dir(type_id):
            if not key.startswith('__'):
                try:
                    setattr(self, key, getattr(self.type_id, key))
                except:
                    pass

    def is_integer(self) -> bool:
        return len(self.shape) == 0 and (self.type_id.kind == 'u' or self.type_id.kind == 'i')

    def is_float(self) -> bool:
        return len(self.shape) == 0 and self.type_id.kind == 'f'

    def is_string(self) -> bool:
        return self.type_id.kind == 'U'

    def is_byte(self) -> bool:
        return self.type_id.kind == 'b' or self.type_id.kind == 'B'

    def is_opaque(self) -> bool:
        return self.type_id.kind == 'O'

    def is_compound(self) -> bool:
        return self.type_id.kind == 'V'

    def is_array(self) -> bool:
        return len(self.shape) > 0


    def get_byte_order(self) -> int:
        match self.type_id.byteorder:
            case '=':
                return 0
            case '<':
                return 0
            case '>':
                return 1
            case '|':
                return 4

    def get_charset(self) -> int:
        """Returns the charset of given np.dtype str.

        Actually the str are unicode objects.
        We return h5py.h5t.CSET_UTF8 by default.
        """
        return 1

    def get_strpad(self) -> int:
        return 0

    def is_variable_str(self) -> bool:
        return True

    def get_metadata(self) -> TypeMetadata:
        base_metadata : TypeMetadata = {
            "class": type(self.type_id).__name__,
            "dtype": str(self),
            "size": self.itemsize if hasattr(self, "itemsize") else (),

        }
        members = {}
        if self.is_integer():
            return {
                **base_metadata,
                "order": self.get_byte_order(),
                "sign": 0 if self.kind == 'u' else 1,
            }

        if self.is_float():
            return {
                **base_metadata,
                "order": self.get_byte_order(),
            }

        if self.is_string():
            return {
                **base_metadata,
                "cset": self.get_charset(),
                "strpad": self.get_strpad(),
                "vlen": self.is_variable_str(),
            }

        if self.is_byte():
            return {**base_metadata, "order": self.get_byte_order()}

        if self.is_opaque():
            return {**base_metadata, "tag": "object"}

        if self.is_compound():
            for name, dt in self.fields.items():
                members[name] = ZarrTypeWrapper(type_id=dt[0], shape=dt[1]).get_metadata()

            return {**base_metadata, "members": members}

        if self.is_array():
            return {
                **base_metadata,
                "dims": self.shape,
                "base": ZarrTypeWrapper(type_id=self.type_id, shape=())
            }

    def __str__(self):
        return str(self.type_id)
