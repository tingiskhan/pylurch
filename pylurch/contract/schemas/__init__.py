from .argument import PatchParser, PutParser, GetParser, PostParser
from .responses import PatchResponse, PutResponse, GetResponse, PostResponse
from ..database import Base
from sqlalchemy import Enum, LargeBinary
from sqlalchemy.orm.attributes import InstrumentedAttribute
from marshmallow_sqlalchemy import SQLAlchemyAutoSchema
from marshmallow import fields as f, ValidationError
from marshmallow_enum import EnumField
from functools import lru_cache


@classmethod
def endpoint(cls):
    if not cls.__name__.endswith("Schema"):
        raise NotImplementedError("All schemas must end with 'Schema'!")

    return cls.__name__.lower().replace("schema", "")


class BytesField(f.Field):
    encoding = 'latin1'

    def _validate(self, value):
        if not isinstance(value, bytes):
            raise ValidationError("Invalid input type!")

        if value is None or value == b"":
            raise ValidationError("Invalid value!")

    def _serialize(self, value, attr: str, obj, **kwargs):
        if value is None:
            return value

        return value.decode(self.encoding)

    def _deserialize(self, value, attr, data, **kwargs):
        if value is None:
            return value

        return value.encode(self.encoding)


def _find_col_types(base, type_):
    return [
            c for c in vars(base).values()
            if isinstance(c, InstrumentedAttribute) and isinstance(c.property.columns[0].type, type_)
        ]


class DatabaseSchema(SQLAlchemyAutoSchema):
    @classmethod
    @lru_cache(maxsize=100)
    def generate_schema(cls, base_class: Base):
        state_dict = {
            "Meta": type("Meta", (object,), {
                "model": base_class,
                "include_fk": True
            }),
            "endpoint": endpoint
        }

        # ===== Custom converters ===== #
        enum_cols = _find_col_types(base_class, Enum)
        byte_cols = _find_col_types(base_class, LargeBinary)

        for ec in enum_cols:
            col = ec.property.columns[0]
            state_dict[col.name] = EnumField(col.type.python_type)

        for bc in byte_cols:
            col = bc.property.columns[0]
            state_dict[col.name] = BytesField(required=False, allow_none=True)

        return type(f"{base_class.__name__}Schema", (DatabaseSchema,), state_dict)

    @classmethod
    def get_schema(cls, obj):
        return cls.generate_schema(obj)

    @classmethod
    def get_subclasses(cls, base: Base):
        res = base.__subclasses__()

        for e in res:
            res.extend(cls.get_subclasses(e))

        return res
