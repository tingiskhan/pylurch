from marshmallow_enum import EnumField
from ..enums import Status
from ..database import Task, TaskMeta
from . import BaseSchema


class TaskSchema(BaseSchema):
    class Meta:
        model = Task

    status = EnumField(Status, required=True)


class TaskMetaSchema(BaseSchema):
    class Meta:
        model = TaskMeta
        include_fk = True