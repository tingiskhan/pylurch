from sqlalchemy.ext.declarative import declarative_base, declared_attr
from sqlalchemy import Column, String, DateTime, func, LargeBinary, Integer, ForeignKey, Enum
from sqlalchemy.orm import relationship
from .enums import ModelStatus, SerializerBackend
from datetime import datetime
import platform
import onnxruntime as rt
import dill
from .schema import ModelSchema


class MyMixin(object):
    @declared_attr
    def __tablename__(cls):
        return cls.__name__

    __mapper_args__ = {
        'always_refresh': True
    }

    id = Column(Integer, primary_key=True)
    upd_at = Column(DateTime, nullable=True, default=func.now())
    upd_by = Column(String(255), nullable=False, default=platform.node(), onupdate=platform.node())
    last_update = Column(DateTime, server_default=func.now(), onupdate=func.now())


Base = declarative_base()


class Model(MyMixin, Base):
    name = Column(String(255), nullable=False, unique=True)

    training_sessions = relationship('TrainingSession', back_populates='model')

    def __init__(self, name):
        self.name = name


class TrainingSession(MyMixin, Base):
    model_id = Column(Integer, ForeignKey(Model.id))
    hash_key = Column(String(255), nullable=False)

    start_time = Column(DateTime, nullable=False)
    end_time = Column(DateTime, nullable=False, default=func.now())

    status = Column(Enum(ModelStatus), nullable=False)

    backend = Column(Enum(SerializerBackend), nullable=False)
    byte_string = Column(LargeBinary(), nullable=True)

    model = relationship(Model, back_populates='training_sessions', uselist=False)
    meta_data = relationship('MetaData')

    def __init__(self, hash_key, start_time, status, backend, end_time=datetime.max, byte_string=None, upd_by=None):
        self.hash_key = hash_key
        self.start_time = start_time
        self.status = status
        self.backend = backend
        self.end_time = end_time
        self.byte_string = byte_string
        self.upd_by = upd_by

    def load(self):
        if self.backend == SerializerBackend.ONNX:
            return rt.InferenceSession(self.byte_string)
        elif self.backend == SerializerBackend.Dill:
            return dill.loads(self.byte_string)

    def to_schema(self):
        out = dict()

        for f in (f_ for f_ in ModelSchema().fields if f_ != 'model_name'):
            out[f] = getattr(self, f)

        out['model_name'] = self.model.name

        return out


class MetaData(MyMixin, Base):
    session_id = Column(Integer, ForeignKey(TrainingSession.id))

    key = Column(String(255), nullable=False)
    value = Column(String(255), nullable=False)

    def __init__(self, key, value):
        self.key = key
        self.value = value