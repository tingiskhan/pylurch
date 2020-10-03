from marshmallow import Schema, fields as f
from marshmallow_enum import EnumField
from ..enums import Status


class Base(Schema):
    status = EnumField(Status)
    message = f.String(required=False)


class GetResponse(Base):
    data = f.String(required=False)
    orient = f.String(required=False)


class PutResponse(Base):
    session_name = f.String(required=True)
    task_id = f.String(required=False)


class PostResponse(Base):
    task_id = f.String(required=True)


class PatchResponse(PutResponse):
    pass