from . import Base, BaseMixin
from sqlalchemy import Column, String, DateTime, LargeBinary, Integer, ForeignKey, Enum
from ..enums import ModelStatus, SerializerBackend
from datetime import datetime


class Model(BaseMixin, Base):
    name = Column(String(255), nullable=False, unique=True)


class TrainingSession(BaseMixin, Base):
    model_id = Column(Integer, ForeignKey(Model.id))
    hash_key = Column(String(255), nullable=False)

    start_time = Column(DateTime, nullable=False, default=datetime.now)
    end_time = Column(DateTime, nullable=False, default=datetime.max)

    status = Column(Enum(ModelStatus), nullable=False)

    backend = Column(Enum(SerializerBackend), nullable=False)
    byte_string = Column(LargeBinary(), nullable=True)


class MetaData(BaseMixin, Base):
    session_id = Column(Integer, ForeignKey(TrainingSession.id))

    key = Column(String(255), nullable=False)
    value = Column(String(255), nullable=False)