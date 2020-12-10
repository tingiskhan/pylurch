from sqlalchemy import Column, String, LargeBinary, Integer, ForeignKey, Enum, UniqueConstraint, select
from sqlalchemy.orm import column_property
from . import Base, BaseMixin
from ..enums import SerializerBackend
from .utils import custom_column_property


class Model(BaseMixin, Base):
    name = Column(String(255), nullable=False, unique=True)


class TrainingResult(BaseMixin, Base):
    session_id = Column(Integer, ForeignKey("TrainingSession.id"), nullable=False, unique=True)
    bytes = Column(LargeBinary(), nullable=False)


class TrainingSession(BaseMixin, Base):
    id = Column(Integer, primary_key=True, autoincrement=True)
    model_id = Column(Integer, ForeignKey(Model.id), nullable=False)
    name = Column(String(255), nullable=False)
    version = Column(Integer(), nullable=False)
    backend = Column(Enum(SerializerBackend, create_constraint=False, native_enum=False), nullable=False)

    has_result = custom_column_property(column_property, "has_result")(
        select([TrainingResult.id]).where(TrainingResult.session_id == id).as_scalar() != None,
        nullable=True,
        default=False
    )

    __table_args__ = (UniqueConstraint(model_id, name, version),)


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


class Package(BaseMixin, Base):
    session_id = Column(Integer, ForeignKey(TrainingSession.id), nullable=False)

    name = Column(String(255), nullable=False)
    version = Column(String(255), nullable=False)
