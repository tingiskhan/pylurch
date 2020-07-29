from marshmallow import Schema, fields as f
from marshmallow_enum import EnumField
from ..enums import Status


class GetResponse(Schema):
    status = EnumField(Status)


class PutResponse(GetResponse):
    session_name = f.String(required=True)
    task_id = f.String(required=False)


class PostResponse(Schema):
    data = f.String(required=True)
    orient = f.String(required=True)


class PatchResponse(PutResponse):
    pass