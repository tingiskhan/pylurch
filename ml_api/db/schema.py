from marshmallow import Schema, fields as f, ValidationError
from marshmallow_enum import EnumField
from .enums import SerializerBackend, ModelStatus


class BytesField(f.Field):
    def _validate(self, value):
        if not isinstance(value, bytes):
            raise ValidationError('Invalid input type!')

        if value is None or value == b'':
            raise ValidationError('Invalid value!')


class ModelSchema(Schema):
    model_name = f.String(required=True)
    hash_key = f.String(required=True)

    start_time = f.DateTime(required=True)
    end_time = f.DateTime()

    upd_by = f.String()

    status = EnumField(ModelStatus, required=True)
    backend = EnumField(SerializerBackend, required=True)

    meta_data = f.Dict(f.String(), f.String(), required=False)

    byte_string = BytesField(required=False)