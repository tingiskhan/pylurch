from sqlalchemy import Column, String, LargeBinary, Integer, ForeignKey, Enum, UniqueConstraint, select, Float
from sqlalchemy.orm import column_property
from . import Base, BaseMixin
from ..enums import SerializerBackend
from .utils import custom_column_property


class Model(BaseMixin, Base):
    name = Column(String(255), nullable=False, unique=True)


class Result(BaseMixin, Base):
    session_id = Column(Integer, ForeignKey("TrainingSession.id"), nullable=False, unique=True)
    backend = Column(Enum(SerializerBackend, create_constraint=False, native_enum=False), nullable=False)
    bytes = Column(LargeBinary(), nullable=False)


class TrainingSession(BaseMixin, Base):
    id = Column(Integer, primary_key=True, autoincrement=True)
    model_id = Column(Integer, ForeignKey(Model.id), nullable=False)
    name = Column(String(255), nullable=False)
    version = Column(Integer(), nullable=False)

    has_result = custom_column_property(column_property, "has_result")(
        select([Result.id]).where(Result.session_id == id).as_scalar() != None, nullable=True, default=False
    )

    __table_args__ = (UniqueConstraint(model_id, name, version),)


class UpdatedSession(BaseMixin, Base):
    base = Column(Integer, ForeignKey(TrainingSession.id), nullable=False)
    new = Column(Integer, ForeignKey(TrainingSession.id), nullable=False)


class Label(BaseMixin, Base):
    session_id = Column(Integer, ForeignKey(TrainingSession.id), nullable=False)
    label = Column(String(), nullable=False)

    __table_args__ = (UniqueConstraint(session_id, label),)


class Score(BaseMixin, Base):
    session_id = Column(Integer, ForeignKey(TrainingSession.id), nullable=False)

    key = Column(String(255), nullable=False)
    value = Column(Float(), nullable=False)

    __table_args__ = (UniqueConstraint(session_id, key),)


class Package(BaseMixin, Base):
    session_id = Column(Integer, ForeignKey(TrainingSession.id), nullable=False)

    name = Column(String(255), nullable=False)
    version = Column(String(255), nullable=False)

    __table_args__ = (UniqueConstraint(session_id, name),)
