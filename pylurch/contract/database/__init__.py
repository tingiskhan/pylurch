from sqlalchemy.ext.declarative import declarative_base, declared_attr
import platform
from sqlalchemy import Column, String, DateTime, Integer
from datetime import datetime


class BaseMixin(object):
    @declared_attr
    def __tablename__(cls):
        return cls.__name__

    __mapper_args__ = {
        'always_refresh': True
    }

    id = Column(Integer, primary_key=True, autoincrement=True)
    upd_at = Column(DateTime, nullable=True, default=datetime.now)
    upd_by = Column(String(255), nullable=False, default=platform.node, onupdate=platform.node)
    last_update = Column(DateTime, default=datetime.now, onupdate=datetime.now)


Base = declarative_base()

SERIALIZATION_IGNORE = tuple(k for (k, v) in vars(BaseMixin).items() if isinstance(v, Column))

from .task import Task, TaskMeta, TaskException
from .inference import MetaData, TrainingSession, Model, UpdatedSession, TrainingResult, Label