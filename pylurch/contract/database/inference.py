from . import Base, BaseMixin
from sqlalchemy import Column, String, LargeBinary, Integer, ForeignKey, Enum, UniqueConstraint, select
from sqlalchemy.orm import column_property
from ..enums import SerializerBackend, Status
from . import Task
from functools import wraps


# TODO: Workaround: https://github.com/tiangolo/pydantic-sqlalchemy/issues/10
def custom_column_property(f, key):
    @wraps(f)
    def wrapper(*args, default=None, nullable=True, **kwargs):
        v = f(*args, **kwargs)
        for c in v.columns:
            c.default = default
            c.nullable = nullable
            # TODO: Perhaps ok...?
            c.key = c.name = key

        return v

    return wrapper


class Model(BaseMixin, Base):
    name = Column(String(255), nullable=False, unique=True)


class TrainingSession(BaseMixin, Base):
    model_id = Column(Integer, ForeignKey(Model.id), nullable=False)
    task_id = Column(Integer, ForeignKey(Task.id), nullable=False, unique=True)
    name = Column(String(255), nullable=False)
    version = Column(Integer(), nullable=False)
    backend = Column(Enum(SerializerBackend, create_constraint=False, native_enum=False), nullable=False)

    status = custom_column_property(column_property, "status")(
        select([Task.status]).where(Task.id == task_id).correlate_except(Task), nullable=False, default=Status.Unknown
    )

    __table_args__ = (UniqueConstraint(model_id, name, version),)


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
