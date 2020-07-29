from marshmallow import fields as f, ValidationError
from marshmallow_enum import EnumField
from ..enums import SerializerBackend
from ..database import Model, TrainingSession, MetaData, UpdatedSession
from . import BaseSchema


class BytesField(f.Field):
    encoding = 'latin1'

    def _validate(self, value):
        if not isinstance(value, bytes):
            raise ValidationError('Invalid input type!')

        if value is None or value == b'':
            raise ValidationError('Invalid value!')

    def _serialize(self, value, attr: str, obj, **kwargs):
        if value is None:
            return value

        return value.decode(self.encoding)

    def _deserialize(self, value, attr, data, **kwargs):
        if value is None:
            return value

        return value.encode(self.encoding)


class ModelSchema(BaseSchema):
    class Meta:
        model = Model


class TrainingSessionSchema(BaseSchema):
    class Meta:
        model = TrainingSession
        include_fk = True

    backend = EnumField(SerializerBackend, required=True)
    byte_string = BytesField(required=False, allow_none=True)


class UpdatedSessionSchema(BaseSchema):
    class Meta:
        model = UpdatedSession
        include_fk = True


class MetaDataSchema(BaseSchema):
    class Meta:
        model = MetaData
        include_fk = True

