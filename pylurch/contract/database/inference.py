from . import Base, BaseMixin
from sqlalchemy import Column, String, LargeBinary, Integer, ForeignKey, Enum, UniqueConstraint
from ..enums import SerializerBackend
from .task import Task


class Model(BaseMixin, Base):
    name = Column(String(255), nullable=False, unique=True)


# TODO: Add version numbering?
class TrainingSession(BaseMixin, Base):
    model_id = Column(Integer, ForeignKey(Model.id), nullable=False)
    task_id = Column(Integer, ForeignKey(Task.id), nullable=False)
    session_name = Column(String(255), nullable=False)
    backend = Column(Enum(SerializerBackend), nullable=False)

    __table_args__ = (
        UniqueConstraint(task_id, session_name),
    )


class TrainingResult(BaseMixin, Base):
    session_id = Column(Integer, ForeignKey(TrainingSession.id), nullable=False, unique=True)
    bytes = Column(LargeBinary(), nullable=True)


class UpdatedSession(BaseMixin, Base):
    base = Column(Integer, ForeignKey(TrainingSession.id), nullable=False)
    new = Column(Integer, ForeignKey(TrainingSession.id), nullable=False)


class Label(BaseMixin, Base):
    session_id = Column(Integer, ForeignKey(TrainingSession.id), nullable=False)
    label = Column(String(), nullable=False)


class MetaData(BaseMixin, Base):
    session_id = Column(Integer, ForeignKey(TrainingSession.id), nullable=False)

    key = Column(String(255), nullable=False)
    value = Column(String(255), nullable=False)