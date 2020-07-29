from . import Base, BaseMixin
from sqlalchemy import Column, String, LargeBinary, Integer, ForeignKey, Enum
from ..enums import SerializerBackend


class Model(BaseMixin, Base):
    name = Column(String(255), nullable=False, unique=True)


# TODO: Add version numbering?
class TrainingSession(BaseMixin, Base):
    model_id = Column(Integer, ForeignKey(Model.id), nullable=False)
    session_name = Column(String(255), nullable=False, unique=True)

    backend = Column(Enum(SerializerBackend), nullable=False)
    byte_string = Column(LargeBinary(), nullable=True)


class UpdatedSession(BaseMixin, Base):
    base = Column(Integer, ForeignKey(TrainingSession.id), nullable=False)
    new = Column(Integer, ForeignKey(TrainingSession.id), nullable=False)


class MetaData(BaseMixin, Base):
    session_id = Column(Integer, ForeignKey(TrainingSession.id), nullable=False)

    key = Column(String(255), nullable=False)
    value = Column(String(255), nullable=False)