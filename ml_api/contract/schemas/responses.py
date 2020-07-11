from marshmallow import Schema, fields as f
from marshmallow_enum import EnumField
from ..enums import ModelStatus


class GetResponse(Schema):
    status = EnumField(ModelStatus)


class PutResponse(GetResponse):
    model_key = f.String(required=True)


class PostResponse(Schema):
    data = f.String(required=True)
    orient = f.String(required=True)


class PatchResponse(GetResponse):
    pass