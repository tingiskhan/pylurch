from . import BaseMixin, Base
from sqlalchemy import Column, String, DateTime, Integer, Enum, ForeignKey
from ..enums import Status
from datetime import datetime
from .exception import ExceptionTemplate


# TODO: Serialize inputs?
class Task(BaseMixin, Base):
    key = Column(String(100), nullable=False, unique=True)

    start_time = Column(DateTime, nullable=False, default=datetime.now)
    end_time = Column(DateTime, nullable=False, default=datetime.max)

    status = Column(Enum(Status, create_constraint=False, native_enum=False), nullable=False)


class TaskMeta(BaseMixin, Base):
    task_id = Column(Integer, ForeignKey(Task.id), nullable=False)

    key = Column(String(100), nullable=False)
    value = Column(String(255), nullable=False)


class TaskException(ExceptionTemplate, Base):
    task_id = Column(Integer, ForeignKey(Task.id), nullable=False)
